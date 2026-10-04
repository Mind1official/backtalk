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
"""The signal bus — tiny files any other program can watch.

The voice line leaves notes; faces read the notes. That one dumb trick
is the whole integration surface:

  .voice_state        idle | listening | thinking | speaking
  .voice_waveform     JSON {ts, samples: [64 floats]} while audio plays
  .voice_loading_pid  exists while the thinking sound is playing
  .voice_rate_limits  JSON {window: {utilization, resets_at}} — only
                      written when show_usage is on
  .voice_context      JSON {used, total, free, pct, categories, ts} —
                      how full the context window is; only written when
                      show_context is on
  .transcript_log     the conversation as plain lines, "YOU: ..." and
                      "<NAME>: ...", appended live as each side speaks

Written to signals_dir (default: the repo root). Visualizers built on
this contract just work.

THE BAREHANDS SEAM: set barehands_state_dir in backtalk.json to a
barehands checkout's state/ folder and the same signals are mirrored in
its format (state/state as a bare word, state/wave.json normalized
0..1) — the on-screen ring becomes your agent's face with zero glue.

Every write is wrapped: the bus must never crash the voice line.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

from backtalk.config import CFG

_DIR = CFG["signals_dir"]
_STATE_FILE = os.path.join(_DIR, ".voice_state")
_WAVEFORM_FILE = os.path.join(_DIR, ".voice_waveform")
_LOADING_PID_FILE = os.path.join(_DIR, ".voice_loading_pid")
_DIRECTION_FILE = os.path.join(_DIR, ".voice_direction")
_REPLY_DONE_FILE = os.path.join(_DIR, ".voice_reply_done")
_RATE_LIMIT_FILE = os.path.join(_DIR, ".voice_rate_limits")
_CONTEXT_FILE = os.path.join(_DIR, ".voice_context")
_PROMPT_FILE = os.path.join(_DIR, ".voice_prompt")
_TYPED_FILE = os.path.join(_DIR, ".voice_typed")
_TRANSCRIPT_FILE = os.path.join(_DIR, ".transcript_log")
_TRANSCRIPT_MAX_LINES = 200
NL = chr(10)  # written through a name so this module never
              # carries a bare escape that a patch tool can mangle

_BH = CFG.get("barehands_state_dir") or ""
_BH_STATE = os.path.join(_BH, "state") if _BH else ""
_BH_WAVE = os.path.join(_BH, "wave.json") if _BH else ""

_THINKING_SOUND = CFG.get("thinking_sound") or ""

_WAVEFORM_MIN_INTERVAL = 1.0 / 15   # ~15 writes/sec is plenty for 60fps reads
_last_waveform_write = 0.0
_static_proc: subprocess.Popen | None = None


# REMOTE TAP: in-process listeners (the remote voice page) hear state
# changes and transcript lines the moment they happen, no file polling.
# A listener must be cheap and thread-safe; one that raises is dropped.
_listeners: list = []
# True while a remote session owns the voice: local speakers and the
# local thinking sound stay silent so the empty room isn't talked at.
REMOTE_MUTE = False


def add_listener(fn):
    _listeners.append(fn)


def _emit(kind: str, data):
    for fn in list(_listeners):
        try:
            fn(kind, data)
        except Exception:
            try:
                _listeners.remove(fn)
            except ValueError:
                pass


def set_state(name: str):
    """Write the state. Never raises — the show must go on."""
    _emit("state", name)
    try:
        with open(_STATE_FILE, "w") as f:
            f.write(name)
    except OSError:
        pass
    if _BH_STATE:
        try:
            with open(_BH_STATE, "w") as f:
                f.write(name)
        except OSError:
            pass


_MUSIC_FILE = os.path.join(_DIR, ".voice_music")


def set_music(on: bool, bands=None):
    """Music mode (music.py): {ts, on, bands}. bands is a 0..1 spectrum
    for the face's visualizer while music plays. The face treats a
    reading older than a few seconds as off, so a voice line that dies
    mid-song can't leave the visualizer up forever. Never raises."""
    _emit("music", {"on": on, "bands": bands})
    try:
        with open(_MUSIC_FILE, "w") as f:
            f.write(json.dumps({"ts": time.time(), "on": bool(on),
                                "bands": bands or []}))
    except OSError:
        pass


def feed_waveform(pcm: np.ndarray):
    """Feed one PCM block (int16) — throttled, downsampled to 64 points.

    Also re-asserts state="speaking" on the same throttle: this only runs
    while the mouth is audibly playing, so the bus self-heals within
    ~70ms if a stray writer stomps the state mid-speech. (That self-heal
    rule once closed a bug that took a whole evening to find.)"""
    global _last_waveform_write
    if pcm.size == 0:
        return
    now = time.time()
    if now - _last_waveform_write < _WAVEFORM_MIN_INTERVAL:
        return
    _last_waveform_write = now
    try:
        idx = np.linspace(0, pcm.size - 1, 64).astype(int)
        raw = pcm[idx].astype(float)
        with open(_WAVEFORM_FILE, "w") as f:
            f.write(json.dumps({"ts": now, "samples": raw.tolist()}))
        if _BH_WAVE:
            norm = np.clip(np.abs(raw) / 32768.0, 0.0, 1.0)
            with open(_BH_WAVE, "w") as f:
                f.write(json.dumps({"ts": now, "samples": norm.tolist()}))
    except (OSError, ValueError):
        pass
    set_state("speaking")


def direction(items):
    """Stage directions the agent wrote into its reply, published at the
    moment the audio carrying them starts playing.

    Your agent can emit `<<anything>>` inline and backtalk will never speak
    it. What the tag MEANS is deliberately not backtalk's business: it
    publishes the raw strings and something else decides. That is the whole
    reason this is a file and not a plugin API.

    The timing is the point, and it is the one part a watcher cannot do for
    itself: these fire when the sentence becomes AUDIBLE, not when the model
    generated it. A screen cue lands on the spoken word instead of seconds
    early. Never raises."""
    if not items:
        return
    try:
        with open(_DIRECTION_FILE, "w") as f:
            f.write(json.dumps({"ts": time.time(), "directions": list(items)}))
    except OSError:
        pass


def reply_done():
    """One reply has finished speaking and its audio has fully drained.

    Distinct from the state going idle, which also happens in the gaps
    BETWEEN sentences of the same reply. Anything waiting for the agent to
    genuinely stop talking wants this rather than a state flicker. Never
    raises."""
    try:
        with open(_REPLY_DONE_FILE, "w") as f:
            f.write(json.dumps({"ts": time.time()}))
    except OSError:
        pass


_rate_limits: dict = {}


def set_rate_limit(window: str, utilization, resets_at):
    """One usage window's reading — how much of the plan is spent.

    Merged rather than replaced, because the reading arrives one window
    at a time and a face wants to draw both at once. `utilization` is a
    0..1 fraction (or None when the window has not reported a number
    yet, which is a real state and not an error); `resets_at` is a unix
    epoch.

    NOTHING CALLS THIS UNLESS show_usage IS ON. That is a privacy
    default, not a performance one: this is the account holder's own
    spend, and it renders on a face that may well be pointed at a
    camera. It never appears without being asked for. (Community fix,
    ai-visualizer issue #1.)

    Never raises."""
    if not window:
        return
    _rate_limits[window] = {"utilization": utilization,
                            "resets_at": resets_at}
    try:
        with open(_RATE_LIMIT_FILE, "w") as f:
            f.write(json.dumps(_rate_limits))
    except OSError:
        pass


def set_context(used, total, categories=None):
    """How full the context window is, for a face to draw.

    `used` and `total` are token counts; `categories` is the CLI's own
    breakdown (a list of {name, tokens}) passed straight through so a
    face can show where the window went. `pct` is precomputed as a 0..1
    fraction because every consumer wants it and none of them should
    have to guess which categories count as occupied.

    Separate from set_rate_limit on purpose. Rate limits are account
    SPEND and stay behind show_usage; this is just how full the current
    conversation is, which leaks nothing about the account. It still has
    its own switch (show_context) so a face pointed at a camera can be
    told to keep quiet about it.

    Never raises."""
    try:
        used = int(used or 0)
        total = int(total or 0)
    except (TypeError, ValueError):
        return
    if total <= 0:
        return
    payload = {"used": used, "total": total,
               "free": max(total - used, 0),
               "pct": round(min(used / total, 1.0), 4),
               "ts": time.time()}
    if categories:
        clean = []
        for c in categories:
            if not isinstance(c, dict):
                continue
            name = str(c.get("name") or "").strip()
            try:
                tok = int(c.get("tokens") or 0)
            except (TypeError, ValueError):
                continue
            if name:
                clean.append({"name": name, "tokens": tok})
        if clean:
            payload["categories"] = clean
    try:
        with open(_CONTEXT_FILE, "w") as f:
            f.write(json.dumps(payload))
    except OSError:
        pass



_transcript_open: str | None = None


def transcript(speaker: str, text: str, append: bool = False):
    """Add a line to the conversation log a face can tail.

    WHY THIS LIVES HERE and not in a Claude Code Stop hook: a hook that
    reads the reply back out of the CLI's transcript file is reading a
    file the CLI has not written yet when the hook fires, so it logs the
    PREVIOUS turn's reply and the panel sits exactly one turn behind
    forever. No amount of waiting fixes it — the write happens after
    hooks complete. The voice line, by contrast, has the text in hand
    the moment it speaks it. The only honest source is the speaker.

    append=True extends the current speaker's last line instead of
    starting a new one, so a reply that streams in sentence by sentence
    reads as one paragraph that grows rather than a stack of fragments.
    Pass append=False (or a different speaker) to start a fresh line.

    Never raises: the bus must never crash the voice line.
    """
    global _transcript_open
    line = " ".join(str(text).split()).strip()
    if not line:
        return
    _emit("transcript", {"speaker": speaker, "text": line, "append": append})
    try:
        lines = []
        if os.path.exists(_TRANSCRIPT_FILE):
            with open(_TRANSCRIPT_FILE, encoding="utf-8",
                      errors="replace") as f:
                lines = f.read().splitlines()
        tag = f"{speaker}:"
        if (append and _transcript_open == speaker and lines
                and lines[-1].startswith(tag)):
            lines[-1] = lines[-1] + " " + line
        else:
            lines.append(f"{tag} {line}")
        _transcript_open = speaker
        with open(_TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
            f.write(NL.join(lines[-_TRANSCRIPT_MAX_LINES:]) + NL)
    except OSError:
        pass


def transcript_end():
    """Close the open line, so the next write starts a new one."""
    global _transcript_open
    _transcript_open = None

def _player_cmd(path: str) -> list[str] | None:
    if sys.platform == "darwin":
        return ["afplay", "-v", "0.12", path]
    for cand in ("ffplay", "aplay", "paplay"):
        from shutil import which
        if which(cand):
            if cand == "ffplay":
                return ["ffplay", "-nodisp", "-autoexit", "-loglevel",
                        "quiet", "-volume", "9", "-af", "lowpass=f=4000",
                        path]
            return [cand, path]
    return None


def static_start():
    """Optional thinking sound — plays while the brain works."""
    global _static_proc
    if REMOTE_MUTE:
        return
    if not _THINKING_SOUND or not os.path.exists(_THINKING_SOUND):
        return
    static_stop()
    cmd = _player_cmd(_THINKING_SOUND)
    if not cmd:
        return
    try:
        _static_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(_LOADING_PID_FILE, "w") as f:
            f.write(str(_static_proc.pid))
    except OSError:
        _static_proc = None


def static_stop():
    global _static_proc
    if _static_proc is not None:
        try:
            _static_proc.terminate()
        except OSError:
            pass
        _static_proc = None
    try:
        os.remove(_LOADING_PID_FILE)
    except OSError:
        pass

# --- typed input from the face (our fork) -----------------------------------
# Two files, one inbound and one outbound, so the agent can ask for something
# typed instead of spoken and get it back through the normal turn pipeline:
#
#   .voice_prompt   WE write it. Non-empty = "an answer is wanted", and the
#                   text is the question. ai-visualizer shows an input box
#                   with that label; empty or absent means no box at all, so
#                   the face stays clean until there is actually a question.
#   .voice_typed    THEY write it. One line per submission, consumed and
#                   truncated by the reader in main.py, which also clears
#                   .voice_prompt so the box disappears the moment it is
#                   answered rather than lingering after the fact.
#
# Deliberately a file pair rather than a socket: the whole bus is files, so
# this needs no new port, no CORS, and it survives either side restarting.


def prompt(text=None):
    """Ask for a typed answer, or clear the request with no argument.

    Safe to call when nothing is listening -- the file simply sits there,
    and a face that is not running cannot miss anything it will not be
    shown later anyway."""
    try:
        if text:
            with open(_PROMPT_FILE, "w", encoding="utf-8") as f:
                f.write(str(text).strip()[:200])
        else:
            try:
                os.remove(_PROMPT_FILE)
            except FileNotFoundError:
                pass
    except OSError:
        pass


def prompt_clear():
    prompt(None)


def take_typed():
    """Lines submitted from a face since the last call, oldest first.

    Reads and TRUNCATES in one pass so a line can never be delivered twice,
    and returns [] on any error -- a missing or half-written file must never
    take down the voice line."""
    try:
        with open(_TYPED_FILE, "r+", encoding="utf-8") as f:
            body = f.read()
            if not body.strip():
                return []
            f.seek(0)
            f.truncate()
    except OSError:
        return []
    return [ln.strip() for ln in body.splitlines() if ln.strip()]
