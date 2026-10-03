# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The warm brain — a persistent Claude session via the Agent SDK,
streaming.

One ClaudeSDKClient lives for the whole voice session: no per-turn
process spawn, no per-turn context reload. Partial-message streaming
means sentences are yielded the moment they're complete, so the mouth
starts speaking while the rest of the thought is still forming.

The session's cwd is YOUR agent's folder (agent_dir in backtalk.json) —
whatever CLAUDE.md lives there defines who is speaking. backtalk adds
only the spoken-delivery discipline (config.DISCIPLINE): the medium,
never the character.
"""
import asyncio
import os
import re
import warnings
from datetime import datetime

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

try:
    from claude_agent_sdk import CanUseToolShadowedWarning
except ImportError:                       # older SDKs: nothing to silence
    CanUseToolShadowedWarning = None

from backtalk import signals
from backtalk.config import CFG, DISCIPLINE
from backtalk.vlog import log

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


SESSION_FILE = os.path.join(CFG["signals_dir"], ".backtalk_session")


class BrainStalled(Exception):
    """A reply went silent past the watchdog limit. The brain has
    already been rebuilt by the time this reaches the caller."""


def _transcript_path(session_id: str) -> str:
    """Where the CLI keeps a session's transcript: one folder per cwd,
    named by swapping every non-alphanumeric character for '-'."""
    root = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.join(
        os.path.expanduser("~"), ".claude")
    slug = re.sub(r"[^A-Za-z0-9]", "-", os.path.abspath(CFG["agent_dir"]))
    return os.path.join(root, "projects", slug, f"{session_id}.jsonl")


def resume_verdict(session_id: str | None) -> tuple[bool, str]:
    """Smart resume: reopen a saved session only while it is light.
    Returns (resume?, reason for the log)."""
    if not session_id:
        return False, "no saved session"
    cap = float(CFG.get("resume_max_mb") or 0)
    try:
        mb = os.path.getsize(_transcript_path(session_id)) / 1048576
    except OSError:
        return False, "saved session transcript not found"
    if cap and mb > cap:
        return False, f"saved session is {mb:.1f} MB (cap {cap:g} MB)"
    return True, f"saved session is {mb:.1f} MB"


def _spoken_tail(n: int = 20) -> list[str]:
    """The last n spoken lines ([you] and the agent's) of the most recent
    voice session in logs/backtalk.log. A session starts at its
    "[backtalk] up" line, so called before this launch logs its own,
    that is the previous session; called mid-session (watchdog), the
    current one."""
    from backtalk.vlog import LOG_PATH
    try:
        with open(LOG_PATH, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()[-3000:]
    except OSError:
        return []
    for i in range(len(lines) - 1, -1, -1):
        if "[backtalk] up" in lines[i]:
            lines = lines[i:]
            break
    name = f"[{CFG.get('name') or 'agent'}]"
    out = []
    for ln in lines:
        body = ln[20:].strip() if len(ln) > 20 else ln.strip()  # drop stamp
        if body.startswith("[you]") or body.startswith(name):
            body = re.sub(r"^(\[[^\]]+\])\s+\(\d+(\.\d+)?s to first\)",
                          r"\1", body)
            out.append(" ".join(body.split())[:220])
    return out[-n:]


def _action_tail(session_id: str | None, n: int = 10) -> list[str]:
    """The agent's last n tool calls, from its OWN transcript (a shared
    activity log would mix in other sessions' work)."""
    if not session_id:
        return []
    import json
    acts = []
    try:
        with open(_transcript_path(session_id), encoding="utf-8",
                  errors="replace") as f:
            for ln in f:
                if '"tool_use"' not in ln or '"assistant"' not in ln:
                    continue
                try:
                    msg = json.loads(ln).get("message") or {}
                except ValueError:
                    continue
                for b in msg.get("content") or []:
                    if not isinstance(b, dict) or b.get("type") != "tool_use":
                        continue
                    inp = b.get("input") or {}
                    what = (inp.get("description") or inp.get("file_path")
                            or inp.get("pattern") or inp.get("url")
                            or inp.get("command") or "")
                    acts.append(f"{b.get('name')}: {' '.join(str(what).split())[:160]}")
    except OSError:
        return []
    return acts[-n:]


def resume_notes(session_id: str | None) -> str:
    """A small briefing for a fresh session that replaced one we did not
    reopen: what was said last and what the agent was doing. Built from
    records that already exist, so it survives a freeze, which never
    gives the agent a chance to write anything itself. Capped at ~4 KB
    so it can't bloat the new session."""
    said, did = _spoken_tail(), _action_tail(session_id)
    parts = []
    if said:
        parts.append("Last things said:\n" + "\n".join(said))
    if did:
        parts.append("Last actions taken:\n" + "\n".join(did))
    text = "\n\n".join(parts) or "(no notes were recoverable)"
    return text[-4000:]


class WarmBrain:
    def __init__(self, model: str | None = None, can_use_tool=None,
                 resume_id: str | None = None):
        # Full model id ON PURPOSE — never a bare alias. The SDK
        # resolves aliases through its own bundled CLI and can silently
        # land on an older model.
        self.model = model or CFG["model"]
        # The spoken permission gate (main.py builds it). Wired at
        # connect in EVERY mode, so a live mode flip needs no reconnect;
        # bypass simply never consults it.
        self._can_use_tool = can_use_tool
        # Session usage, spoken on request ("usage report").
        self.session = {"turns": 0, "out_tokens": 0, "in_tokens": 0,
                        "cost": 0.0}
        self._client: ClaudeSDKClient | None = None
        # The session to reattach to at the FIRST start only (config key
        # resume_last_session). Consumed on use: a desync rebuild in
        # reset_turn() must always start FRESH: a rebuild means a turn
        # went sideways mid-stream, the wrong moment to gamble on
        # reattaching. (Community proposal, issue #1.)
        self._resume_id = resume_id
        # The live session's id (from every ResultMessage), so a watchdog
        # rebuild can reattach to THIS conversation rather than lose it.
        self.session_id: str | None = None
        # Set when a fresh session replaced one we chose not to reopen:
        # the next query carries a one-line pointer to the agent's notes.
        self._notes_hint: str | None = None
        # True while a query's response hasn't been consumed through its
        # ResultMessage — i.e. the shared message pipe may hold leftovers.
        self._dirty = False

    async def start(self):
        mode = CFG["permission_mode"]
        if mode == "default":
            mode = "ask"     # legacy alias, see config.py
        # backtalk's "ask" = the SDK's "default" mode with gated calls
        # routed to the spoken can_use_tool gate.
        sdk_mode = "default" if mode == "ask" else mode
        if sdk_mode == "bypassPermissions" and self._can_use_tool \
                and CanUseToolShadowedWarning:
            # Deliberate auto-approve: the SDK warns that the callback is
            # shadowed. That IS the chosen behavior, so boot quietly.
            warnings.filterwarnings("ignore",
                                    category=CanUseToolShadowedWarning)
        resume, self._resume_id = self._resume_id, None   # consume once

        def _opts(rid):
            return ClaudeAgentOptions(
                cwd=CFG["agent_dir"],
                model=self.model,
                system_prompt={"type": "preset", "preset": "claude_code",
                               "append": DISCIPLINE},
                include_partial_messages=True,
                permission_mode=sdk_mode,
                can_use_tool=self._can_use_tool,
                add_dirs=CFG["extra_dirs"],
                skills=CFG["visible_skills"],
                resume=rid,
            )
        if resume:
            try:
                self._client = ClaudeSDKClient(options=_opts(resume))
                await self._client.connect()
                log(f"[brain] resumed session {resume[:8]}")
                return
            except Exception as e:
                # a stale or invalid saved session must never brick the
                # launch. Fall back to a fresh conversation and say so.
                log(f"[brain] resume failed ({str(e)[:80]}), "
                    f"starting fresh")
                try:
                    await self._client.disconnect()
                except Exception:
                    pass
        self._client = ClaudeSDKClient(options=_opts(None))
        await self._client.connect()

    async def set_permission_mode(self, backtalk_mode: str):
        """Live flip, no reconnect, conversation intact ("ask" maps to
        the SDK's "default", whose gated calls hit the spoken gate)."""
        if self._client:
            sdk_mode = "default" if backtalk_mode == "ask" \
                else backtalk_mode
            await self._client.set_permission_mode(sdk_mode)

    async def context_usage(self):
        """The CLI's own context-window breakdown, or None."""
        try:
            return await self._client.get_context_usage()
        except Exception:
            return None

    async def _publish_context(self):
        """Put the context-window fill on the signal bus for the face.

        The CLI's breakdown is the only honest source for the window
        SIZE: it reports every category including "Free space" and the
        autocompact buffer, so the categories sum to the whole window
        while the occupied ones sum to what we are actually holding.
        That is why total is a sum and not a constant — the window can
        change under us with a model or setting change, and a hardcoded
        number would quietly start lying.

        Bounded and fully swallowed, like the rate-limit pull: a face
        readout must never cost a turn. If this stops working the meter
        just goes stale, and that is the intended failure.

        Never raises."""
        if not CFG.get("show_context"):
            return
        try:
            ctx = await asyncio.wait_for(self.context_usage(), 5)
            cats = (getattr(ctx, "categories", None)
                    or (ctx or {}).get("categories") or [])
            cats = [c for c in cats if isinstance(c, dict)]
            if not cats:
                return

            def _tok(c):
                try:
                    return int(c.get("tokens") or 0)
                except (TypeError, ValueError):
                    return 0

            def _spare(c):
                n = str(c.get("name", "")).lower()
                return "free" in n or "buffer" in n

            total = sum(_tok(c) for c in cats)
            used = sum(_tok(c) for c in cats if not _spare(c))
            signals.set_context(used, total, cats)
        except Exception:
            pass

    def _remember_session(self, rm):
        """Persist the session id after a completed turn, so the next
        launch can reattach (config: resume_last_session). Must never
        break a turn; silence on any failure."""
        sid = getattr(rm, "session_id", None)
        if not sid:
            return
        self.session_id = sid
        if not CFG.get("resume_last_session"):
            return
        try:
            with open(SESSION_FILE, "w") as f:
                f.write(sid)
        except OSError:
            pass

    def _tally(self, rm, count_turn=True):
        """Session usage bookkeeping. Must never break a turn."""
        try:
            u = getattr(rm, "usage", None) or {}
            s = self.session
            if count_turn:
                s["turns"] += 1
            s["out_tokens"] += int(u.get("output_tokens") or 0)
            s["in_tokens"] += (int(u.get("input_tokens") or 0)
                               + int(u.get("cache_read_input_tokens")
                                     or 0))
            c = getattr(rm, "total_cost_usd", None)
            if c:
                s["cost"] += float(c)
        except Exception:
            pass

    async def _pull_rate_limits(self):
        """Ask the CLI outright how much of the plan is spent.

        A DIRECT QUERY, not the RateLimitEvent stream. The event fires
        rarely and usually arrives carrying resets_at with no utilization
        at all, so a listener built on it reports nothing most of the
        time -- which is exactly how this feature looked broken for its
        whole life. (Community fix, ai-visualizer issue #1.)

        THIS REACHES PAST THE SDK'S PUBLIC SURFACE ON PURPOSE, and a
        reader should know it rather than discover it. `get_usage` is a
        control request the bundled CLI answers but the SDK never wraps,
        so there is no supported call to make. The supported-looking
        alternative is a dead end and was tested as one: the terminal
        status line never fires in a headless session, so its numbers
        are unreachable from here.

        Which means this can stop working without anyone doing anything
        wrong, and the containment is the point. Every failure is
        swallowed and the readout simply goes quiet. It must never cost
        a turn, so it is also bounded -- an unanswered control request
        would otherwise hang the voice line mid-conversation."""
        if not CFG.get("show_usage"):
            return
        try:
            usage = await asyncio.wait_for(
                self._client._query._send_control_request(
                    {"subtype": "get_usage"}), 5)
            for window in ("five_hour", "seven_day"):
                w = (usage.get("rate_limits") or {}).get(window)
                if not w:
                    continue
                # Two spellings accepted deliberately: this shape is not
                # documented anywhere, so the cheap tolerance is worth
                # more than the tidiness. Both are percentages, and the
                # rest of the pipeline wants a 0..1 fraction.
                pct = w.get("utilization")
                if pct is None:
                    pct = w.get("used_percentage")
                pct = pct / 100 if pct is not None else None
                resets = w.get("resets_at")
                if isinstance(resets, str):
                    resets = int(datetime.fromisoformat(resets).timestamp())
                signals.set_rate_limit(window, pct, resets)
        except Exception:
            pass

    async def command(self, cmd: str) -> str:
        """Run a console slash command (/clear, /compact, /model,
        /effort) through the normal stream and return whatever text the
        CLI answered with (confirmations, errors). Slash-command replies
        arrive as COMPLETE AssistantMessages, not stream deltas, so
        ask_stream cannot see them. Bounded like reset_turn is: this
        stream is not trusted to always deliver, and an unbounded await
        here would deafen the whole voice loop. On timeout the pipe is
        left marked dirty so the next reset_turn drains or rebuilds."""
        self._dirty = True
        await self._client.query(cmd)
        texts = []

        async def _collect():
            async for msg in self._client.receive_response():
                t = type(msg).__name__
                if t == "AssistantMessage":
                    for b in getattr(msg, "content", []) or []:
                        txt = getattr(b, "text", None)
                        if txt:
                            texts.append(txt)
                elif t == "ResultMessage":
                    self._dirty = False
                    self._tally(msg, count_turn=False)
                    self._remember_session(msg)
                    break

        try:
            await asyncio.wait_for(_collect(), 90)
        except asyncio.TimeoutError:
            log(f"[brain] console command timed out: {cmd!r}")
            return "error: the command timed out"
        return " ".join(texts).strip()

    async def interrupt(self):
        if self._client:
            await self._client.interrupt()

    async def reset_turn(self, timeout: float = 8.0):
        """Re-align the message pipe after an interrupted/failed turn.

        THE OFF-BY-ONE BUG, and why this method exists: the SDK client
        has ONE shared message stream and receive_response() stops at
        the FIRST ResultMessage it sees — there is no pairing between a
        query and its response. A cancelled turn stops consuming
        mid-stream, leaving the dead turn's remaining messages
        (including its ResultMessage) buffered. The next query then
        pairs with those leftovers: the first ask lands on the stale
        ResultMessage and yields nothing, and every ask after that
        answers the PREVIOUS question — for the rest of the session.
        So: interrupt the dead turn, then drain the pipe through its
        stale ResultMessage before the next query goes out. No-op when
        the last turn was consumed clean."""
        if not self._client or not self._dirty:
            return
        try:
            await asyncio.wait_for(self._client.interrupt(), 5)
        except Exception:
            pass  # turn may already be over — the drain below is the point

        async def _drain() -> int:
            n = 0
            async for msg in self._client.receive_response():
                n += 1
                if type(msg).__name__ == "ResultMessage":
                    break
            return n

        try:
            drained = await asyncio.wait_for(_drain(), timeout)
            log(f"[brain] interrupted turn drained ({drained} stale messages)")
            self._dirty = False
        except Exception:
            # Can't re-align — rebuild the session rather than run
            # desynced. Loses this voice session's conversation memory;
            # better than answering every question one turn late for the
            # rest of the day.
            log("[brain] stream desynced beyond repair — rebuilding the "
                "session (conversation memory for this session resets)")
            try:
                await self._client.disconnect()
            except Exception:
                pass
            self._client = None
            await self.start()
            self._dirty = False

    async def stop(self):
        if self._client:
            await self._client.disconnect()
            self._client = None

    def skipped_resume(self, reason: str, session_id: str | None = None):
        """A fresh session replaced one we chose not to reopen (smart
        resume at launch, or a watchdog rebuild): brief its first turn
        on where things stood, so it doesn't start blind."""
        notes = resume_notes(session_id)
        self._notes_hint = (
            f"[backtalk: the previous voice conversation was not reopened "
            f"({reason}). Resume notes from it follow. Use them to pick up "
            f"where things left off; check today's daily note for anything "
            f"older.]\n{notes}\n[end of resume notes]")

    async def _rebuild_after_stall(self):
        """Tear down the hung CLI and stand the brain back up. Reattaches
        to this same conversation while it is light enough (the hung
        request never reached the transcript); a heavy one starts fresh
        with the notes pointer instead. Never raises past here."""
        sid = self.session_id
        try:
            await asyncio.wait_for(self._client.disconnect(), 10)
        except Exception:
            pass
        self._client = None
        ok, why = resume_verdict(sid)
        self._resume_id = sid if ok else None
        log(f"[watchdog] rebuilding the brain: "
            f"{'reattaching' if ok else 'starting fresh'} ({why})")
        if not ok and sid:
            self.skipped_resume(why, sid)
        await self.start()
        self._dirty = False

    async def ask_stream(self, utterance: str):
        """Yield complete sentences as they stream out of the model.

        THE WATCHDOG: every wait for the next message is bounded. A model
        request can hang with the connection open and no error (seen
        twice in two days), and an unbounded wait strands the voice line
        in silence forever. Text and thinking stream in constantly, so
        quiet only means trouble when no tool is running; a running tool
        gets the long limit. On a stall the brain is rebuilt and
        BrainStalled is raised for the caller to say so out loud."""
        self._dirty = True             # in flight until its ResultMessage
        if self._notes_hint:
            utterance, self._notes_hint = (
                f"{self._notes_hint}\n\n{utterance}", None)
        await self._client.query(utterance)
        buf = ""
        quiet = float(CFG.get("stall_timeout_s") or 0) or None
        tool_quiet = float(CFG.get("tool_stall_timeout_s") or 0) or None
        tools_open = 0                 # tool_use blocks without a result yet
        stream = self._client.receive_response().__aiter__()
        while True:
            limit = tool_quiet if tools_open > 0 else quiet
            try:
                msg = await asyncio.wait_for(stream.__anext__(), limit)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                log(f"[watchdog] no word from the brain in {limit:.0f}s "
                    f"({'tool running' if tools_open else 'no tool running'})"
                    f" — declaring the turn stalled")
                await self._rebuild_after_stall()
                raise BrainStalled()
            t = type(msg).__name__
            if t in ("AssistantMessage", "UserMessage"):
                blocks = getattr(msg, "content", None)
                if isinstance(blocks, list):
                    for b in blocks:
                        n = type(b).__name__
                        if n == "ToolUseBlock":
                            tools_open += 1
                        elif n == "ToolResultBlock":
                            tools_open = max(0, tools_open - 1)
            if t == "StreamEvent":
                ev = getattr(msg, "event", {}) or {}
                if ev.get("type") == "content_block_delta":
                    delta = ev.get("delta", {}) or {}
                    if delta.get("type") == "text_delta":
                        buf += delta.get("text", "")
                        # emit any complete sentences
                        while True:
                            m = _SENTENCE_END.search(buf)
                            if not m:
                                break
                            sentence, buf = (buf[:m.end()].strip(),
                                             buf[m.end():])
                            if sentence:
                                yield sentence
                elif ev.get("type") == "content_block_stop":
                    # End of a speech block (e.g. right before a tool
                    # call): flush NOW. Without this, pre-tool filler
                    # ("On it — let me grab that.") sits silent in the
                    # buffer through the whole tool run, then plays
                    # glued to the answer: long dead air, then two
                    # thoughts at once.
                    tail = buf.strip()
                    buf = ""
                    if tail:
                        yield tail
            elif t == "ResultMessage":
                self._dirty = False    # turn fully consumed — pipe aligned
                self._tally(msg)
                self._remember_session(msg)
                await self._pull_rate_limits()
                await self._publish_context()
                break
        tail = buf.strip()
        if tail:
            yield tail


if __name__ == "__main__":
    import time

    async def demo():
        b = WarmBrain()
        await b.start()
        for prompt in ("Voice check: greet me in one sentence.",
                       "And what's two plus two, spoken like yourself?"):
            t0 = time.time()
            async for s in b.ask_stream(prompt):
                print(f"  ({time.time()-t0:4.1f}s) {s}", flush=True)
        await b.stop()

    asyncio.run(demo())
