"""Remote voice: talk to the agent from a phone or tablet browser.

A small web server (aiohttp, same event loop as the voice line) serves a
hold-to-talk page and one WebSocket per phone:

  phone -> server   JSON {"t":"press"} when a finger lands (barge-in),
                    then one binary message: the recorded clip (any
                    container ffmpeg reads: webm/opus, mp4/aac)
                    JSON {"t":"auth","token":...} to resume a session
  server -> phone   JSON events: hello, state, transcript, heard, auth,
                    locked, flush, notice
                    binary: 4-byte little-endian sample rate + int16 PCM
                    (the agent's voice, paced at real time by the mouth)

TWO LOCKS, ON PURPOSE. A spoken passcode alone is weak: it can leak on a
stream VOD, a room mic, or a replayed recording, and whoever gets in
drives an agent on this PC. So:
  1. Cloudflare Access sits in front of the public hostname (a login),
     and every request that arrives through Cloudflare must carry a
     valid Access token (verified in access.py) or it is refused.
  2. The spoken passcode, checked here against a salted PBKDF2 hash.
     Three misses lock remote access for 15 minutes. An unlock lasts
     until 30 idle minutes pass.
The server binds 127.0.0.1 only; cloudflared is the only way in from
outside.

WHILE A REMOTE SESSION IS LIVE: the PC speakers and thinking sound are
muted (signals.REMOTE_MUTE + mouth.remote), the local open mic yields,
and permissions drop to "ask" (main.py's on_begin/on_end), so anything
real gets a spoken "can I" answered from the phone. Auto-approve never
leaves the house.
"""
import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from backtalk import signals
from backtalk.config import CFG
from backtalk.vlog import log

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_FAILS = 3
LOCKOUT_S = 15 * 60
IDLE_S = 30 * 60
GRACE_S = 90          # a dropped connection (phone screen locked) keeps
                      # the session this long before handing back to local
MAX_CLIP_BYTES = 8 * 1024 * 1024

STATE = {"active": False}   # main.py's mic gate reads this


# ----------------------------------------------------------- passcode --

def _norm(text: str) -> str:
    """What the passcode comparison sees: lowercase words, no
    punctuation. Whisper punctuates and capitalizes freely."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def hash_passcode(phrase: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", _norm(phrase).encode(), salt, 200_000)
    return f"pbkdf2${salt.hex()}${dk.hex()}"


def check_passcode(heard: str, stored: str) -> bool:
    try:
        _, salt, want = stored.split("$")
    except ValueError:
        return False
    got = hash_passcode(heard, bytes.fromhex(salt)).split("$")[2]
    return hmac.compare_digest(got, want)


# ------------------------------------------------------------- audio ----

def _decode(blob: bytes) -> np.ndarray:
    """Any recorded clip -> int16 mono 16 kHz. Via a temp file, not a
    pipe: Safari's mp4 keeps its index at the END, which a pipe can't
    seek back to."""
    fd, path = tempfile.mkstemp(suffix=".clip")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
        r = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", path, "-f", "s16le",
             "-ac", "1", "-ar", "16000", "pipe:1"],
            capture_output=True, timeout=30)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode(errors="replace")[:200])
        return np.frombuffer(r.stdout, dtype=np.int16)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# ------------------------------------------------------------- server ---

class Remote:
    def __init__(self, loop, mouth, typed_q, on_begin, on_end, quit_phrases,
                 on_bypass=None):
        self.loop = loop
        self.mouth = mouth
        self.typed_q = typed_q
        self.on_begin = on_begin        # async () -> None
        self.on_end = on_end            # async () -> None
        # async () -> bool. Flips the LIVE session to bypass and returns
        # whether it actually took. False means keep asking: a bypass
        # that silently failed would be the worst of both worlds.
        self.on_bypass = on_bypass
        # Per-session, and that is the whole point of the lock: a new
        # remote session always starts gated, whatever the last one did.
        self.bypassed = False
        self.quit_phrases = [_norm(q) for q in quit_phrases]
        self.ws = None                  # the one live phone
        self.out_q: asyncio.Queue = asyncio.Queue()
        self.token = None
        self.last_seen = 0.0
        self.fails = 0
        self.locked_until = 0.0
        self.busy = False               # one clip at a time
        signals.add_listener(self._on_signal)

    # --- mouth sink (called from the mouth's worker thread) ---
    def active(self) -> bool:
        return STATE["active"]

    def send_audio(self, rate: int, pcm: np.ndarray):
        self._post(struct.pack("<I", rate) + pcm.astype("<i2").tobytes())

    def flush(self):
        self._post({"t": "flush"})

    def _on_signal(self, kind, data):
        if self.ws is not None:
            self._post({"t": kind, "data": data})

    def _post(self, item):
        try:
            self.loop.call_soon_threadsafe(self.out_q.put_nowait, item)
        except RuntimeError:
            pass                        # loop closed: shutting down

    async def _sender(self):
        """The single writer, so audio frames never interleave."""
        while True:
            item = await self.out_q.get()
            ws = self.ws
            if ws is None or ws.closed:
                continue
            try:
                if isinstance(item, bytes):
                    await ws.send_bytes(item)
                else:
                    await ws.send_str(json.dumps(item))
            except Exception:
                pass

    # --- session lifecycle ---
    async def _begin(self):
        if STATE["active"]:
            return
        STATE["active"] = True
        signals.REMOTE_MUTE = True
        signals.set_remote(True)
        signals.static_stop()
        self.mouth.shut_up()            # nothing half-said at the desk
        self.mouth.remote = self
        self.bypassed = False           # every takeover starts gated
        log("[remote] session live: PC speakers muted, local mic paused")
        await self.on_begin()

    async def _end(self, why: str):
        self.token = None
        if not STATE["active"]:
            return
        STATE["active"] = False
        signals.REMOTE_MUTE = False
        signals.set_remote(False)
        self.mouth.shut_up()
        self.mouth.remote = None
        self.bypassed = False
        log(f"[remote] session ended ({why}): back to local")
        await self.on_end()

    async def end_now(self, why: str = "taken back at the desk"):
        """End a live session from the PC side and put the phone's keypad
        back up. The desk always outranks the phone: whoever is physically
        at the machine gets the final say over who holds the line."""
        await self._end(why)
        await self._hello()

    async def _watch(self):
        """Ends an idle session, or one whose phone never came back."""
        while True:
            await asyncio.sleep(10)
            if not STATE["active"]:
                continue
            idle = time.time() - self.last_seen
            if idle > IDLE_S:
                await self._end("idle 30 minutes")
            elif self.ws is None and idle > GRACE_S:
                await self._end("phone disconnected")

    # --- HTTP ---
    def _from_outside(self, request) -> bool:
        return "Cf-Connecting-Ip" in request.headers

    async def _admit(self, request) -> bool:
        if not self._from_outside(request):
            return True                 # this PC (bound to 127.0.0.1)
        from backtalk.access import verify
        ok, who = await verify(request.headers.get("Cf-Access-Jwt-Assertion"))
        if not ok:
            log(f"[remote] refused an outside request: {who}")
        return ok

    async def page(self, request):
        from aiohttp import web
        if not await self._admit(request):
            return web.Response(status=403, text="Forbidden")
        return web.FileResponse(WEB_DIR / "remote.html",
                                headers={"Cache-Control": "no-store"})

    async def socket(self, request):
        from aiohttp import web
        if not await self._admit(request):
            return web.Response(status=403, text="Forbidden")
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=MAX_CLIP_BYTES)
        await ws.prepare(request)
        if self.ws is not None and not self.ws.closed:
            await self.ws.send_str(json.dumps(
                {"t": "notice", "data": "Another device took over."}))
            await self.ws.close()
        self.ws = ws
        where = request.headers.get("Cf-Connecting-Ip", "this PC")
        log(f"[remote] phone connected from {where}")
        await self._hello()
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    await self._on_text(json.loads(msg.data))
                elif msg.type == web.WSMsgType.BINARY:
                    await self._on_clip(msg.data)
        finally:
            if self.ws is ws:
                self.ws = None
            log("[remote] phone disconnected")
        return ws

    async def _hello(self):
        await self._send({"t": "hello", "data": {
            "name": CFG.get("name") or "agent",
            "authed": False,
            "locked": max(0, int(self.locked_until - time.time())),
            "has_passcode": bool(CFG.get("remote_passcode_hash"))}})

    async def _send(self, obj):
        self._post(obj)

    def post_ask(self, text: str | None) -> None:
        """Put a pending permission question on the phone as a card with
        Approve/Deny, or clear it with None. Thread-safe like the rest of
        the outbound side: it goes through the same send queue.

        The phone's buttons answer through the EXISTING typed path (they
        send the literal word "yes"/"no"), so the gate in main.py needs
        no second answering mechanism -- one place decides, one place
        normalizes, and a tap and a spoken yes cannot disagree."""
        if self.active():
            self._post({"t": "ask", "data": text})

    async def _on_text(self, m):
        t = m.get("t")
        if t == "auth":
            tok = m.get("token") or ""
            if (self.token and hmac.compare_digest(tok, self.token)
                    and time.time() - self.last_seen < IDLE_S):
                self.last_seen = time.time()
                await self._begin()
                await self._send({"t": "auth", "data": {"ok": True}})
            else:
                await self._send({"t": "auth", "data": {"ok": False}})
        elif t == "pin":
            # Deliberately NOT gated on "not STATE[active]". A window
            # closed mid-session takes its resume token with it, and the
            # session outlives it for the grace period -- so refusing the
            # PIN here left no way back in AND no way for the desk to
            # speak, because the local mic stays paused while a session
            # is live. The PIN is still required either way: _gate holds
            # the one attempt/lockout path and _begin is a no-op when a
            # session is already up, so this reclaims a session, never
            # opens an unlocked one.
            now = time.time()
            if now < self.locked_until:
                await self._send({"t": "locked",
                                  "data": int(self.locked_until - now)})
                return
            await self._gate(str(m.get("data") or "")[:32])
        elif t == "press" and STATE["active"]:
            self.last_seen = time.time()
            if self.mouth.speaking:
                self.mouth.shut_up()    # barge-in: a finger on the glass
        elif t == "answer" and STATE["active"]:
            # A tapped Approve/Deny. It is fed in as plain words on the
            # same queue as a typed line, so the spoken gate's own
            # vocabulary and its deny-by-default rule still decide.
            # An explicit allowlist of the three words the page can send.
            # This was "yes" if data == "yes" else "no", which silently
            # turned the THIRD button -- "YES, AND STOP ASKING FOR THIS",
            # which sends "always" -- into a DENY. A yes button that means
            # no is the one failure a permission gate must never have, and
            # it quoted the person as saying "no" when they never did.
            # Still deny-by-default: anything unrecognised becomes "no".
            raw = str(m.get("data") or "").strip().lower()
            word = raw if raw in ("yes", "no", "always") else "no"
            self.last_seen = time.time()
            log(f"[remote] permission answered by tap: {word}")
            self.typed_q.put(word)
        elif t == "bypass":
            # THE SECOND LOCK, REUSED ON PURPOSE. Turning the gate off
            # for a remote session is at least as serious as opening one,
            # so it is held to the SAME passcode, the same three-strike
            # counter and the same 15-minute lockout -- not a confirm
            # tap, which a hijacked live session would sail straight
            # through. No passcode set on the PC means no bypass at all.
            #
            # Deliberately gated on a LIVE session: there is no session
            # to bypass otherwise, and allowing it earlier would let the
            # passcode turn the gate off before anyone took the floor.
            if not STATE["active"]:
                await self._send({"t": "bypass",
                                  "data": {"ok": False,
                                           "why": "No session is live."}})
                return
            now = time.time()
            if now < self.locked_until:
                await self._send({"t": "locked",
                                  "data": int(self.locked_until - now)})
                return
            stored = CFG.get("remote_passcode_hash") or ""
            if not stored:
                await self._send({"t": "bypass", "data": {
                    "ok": False,
                    "why": "No passcode is set on the PC."}})
                return
            heard = str(m.get("data") or "")[:64]
            if not (heard and check_passcode(heard, stored)):
                self.fails += 1
                log(f"[remote] bypass passcode rejected "
                    f"({self.fails}/{MAX_FAILS})")
                if self.fails >= MAX_FAILS:
                    self.fails = 0
                    self.locked_until = time.time() + LOCKOUT_S
                    log("[remote] LOCKED for 15 minutes after 3 bad "
                        "passcodes")
                    await self._send({"t": "locked", "data": LOCKOUT_S})
                else:
                    await self._send({"t": "bypass", "data": {
                        "ok": False, "left": MAX_FAILS - self.fails,
                        "why": "That passcode did not match."}})
                return
            self.fails = 0
            self.last_seen = time.time()
            took = False
            if self.on_bypass is not None:
                try:
                    took = bool(await self.on_bypass())
                except Exception as e:
                    log(f"[remote] bypass hook failed: {e}")
                    took = False
            self.bypassed = took
            # SAID OUT LOUD EITHER WAY. A gate that turns itself off
            # quietly is indistinguishable from one that was never on.
            if took:
                log("[remote] BYPASS ON for this remote session")
                self.mouth.say("Permission checks are off for this "
                               "remote session. They come back on when "
                               "the session ends.")
            else:
                log("[remote] bypass refused: the mode did not flip")
                self.mouth.say("I could not turn the checks off, so "
                               "I'll keep asking.")
            await self._send({"t": "bypass", "data": {
                "ok": took,
                "why": None if took
                       else "The mode would not flip, so it still asks."}})
        elif t == "bye":
            await self._end("signed off on the phone")
            await self._hello()

    async def _on_clip(self, blob: bytes):
        if self.busy:
            return
        self.busy = True
        try:
            await self._handle_clip(blob)
        except Exception as e:
            log(f"[remote] clip failed: {e}")
            await self._send({"t": "notice", "data": "That clip didn't come through. Try again."})
        finally:
            self.busy = False

    async def _gate(self, text: str) -> None:
        """One attempt at the gate, from a spoken clip or a typed PIN.
        Both normalize through _norm, so the same stored hash answers
        either: a PIN is just a passcode made of digits."""
        stored = CFG.get("remote_passcode_hash") or ""
        if not stored:
            await self._send({"t": "notice",
                              "data": "No passcode is set on the PC yet."})
            return
        if text and check_passcode(text, stored):
            self.fails = 0
            self.token = secrets.token_urlsafe(32)
            self.last_seen = time.time()
            log("[remote] passcode accepted")
            await self._send({"t": "auth",
                              "data": {"ok": True, "token": self.token}})
            await self._begin()
            return
        self.fails += 1
        log(f"[remote] passcode rejected ({self.fails}/{MAX_FAILS})")
        if self.fails >= MAX_FAILS:
            self.fails = 0
            self.locked_until = time.time() + LOCKOUT_S
            log("[remote] LOCKED for 15 minutes after 3 bad passcodes")
            await self._send({"t": "locked", "data": LOCKOUT_S})
        else:
            await self._send({"t": "auth", "data": {
                "ok": False, "left": MAX_FAILS - self.fails}})

    async def _handle_clip(self, blob: bytes):
        from backtalk.ears import transcribe
        now = time.time()
        if now < self.locked_until:
            await self._send({"t": "locked",
                              "data": int(self.locked_until - now)})
            return
        pcm = await self.loop.run_in_executor(None, _decode, blob)
        if pcm.size < 1600:             # under 0.1 s: a mis-tap
            return
        text = (await self.loop.run_in_executor(None, transcribe, pcm)).strip()

        if not STATE["active"]:                     # ---- the gate ----
            await self._gate(text)
            return

        self.last_seen = time.time()                # ---- live ----
        if not text:
            await self._send({"t": "notice", "data": "Didn't catch that."})
            return
        if any(q and q in _norm(text) for q in self.quit_phrases):
            # A quit phrase from the phone ends the REMOTE session only;
            # it must never hang up the whole voice line at home.
            await self._end("quit phrase from the phone")
            await self._hello()
            return
        await self._send({"t": "heard", "data": text})
        self.typed_q.put(text)          # the same path as a typed line


async def start(loop, mouth, typed_q, on_begin, on_end, quit_phrases,
                on_bypass=None):
    """Start the remote server inside the voice line's event loop.
    Returns the Remote, or None when remote is off or fails to bind."""
    if not CFG.get("remote_enabled"):
        return None
    from aiohttp import web
    r = Remote(loop, mouth, typed_q, on_begin, on_end, quit_phrases,
               on_bypass)
    app = web.Application(client_max_size=MAX_CLIP_BYTES)
    app.router.add_get("/", r.page)
    app.router.add_get("/ws", r.socket)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    port = int(CFG.get("remote_port") or 8797)
    try:
        await web.TCPSite(runner, "127.0.0.1", port).start()
    except OSError as e:
        log(f"[remote] could not bind 127.0.0.1:{port} ({e}); remote is off")
        return None
    loop.create_task(r._sender())
    loop.create_task(r._watch())
    log(f"[remote] listening on http://127.0.0.1:{port}/"
        + ("" if CFG.get("remote_passcode_hash")
           else " (NO PASSCODE SET: run python -m backtalk.remote --set-passcode)"))
    return r


def _set_passcode():
    """Typed by the owner at the PC, never echoed, stored only as a hash."""
    import getpass
    from backtalk.config import CONFIG_PATH
    print(f"Writing to: {_config_arg() or CONFIG_PATH}")
    print("Choose a spoken passcode: several plain words, no numbers, "
          "nothing you ever say on stream.")
    a = getpass.getpass("Passcode: ")
    b = getpass.getpass("Again: ")
    if _norm(a) != _norm(b) or len(_norm(a).split()) < 3:
        print("Not set: the two didn't match, or it was under three words.")
        return 1
    print(f"Passcode set in {_write_hash(a)}. Restart the voice line to use it.")
    return 0


def _config_arg() -> str | None:
    """`--config <path>`: write the hash to THIS file, whatever the
    environment says. Learned the hard way -- BACKTALK_CONFIG set in a
    spawned console did not reach the child, and a PIN meant for a
    second instance silently overwrote the first instance's."""
    if "--config" in sys.argv:
        try:
            return sys.argv[sys.argv.index("--config") + 1]
        except IndexError:
            return None
    return None


def _write_hash(phrase: str) -> str:
    """Store a new gate secret as a salted hash. Returns the path."""
    from backtalk.config import CONFIG_PATH
    CONFIG_PATH = _config_arg() or CONFIG_PATH
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        cfg = {}
    cfg["remote_passcode_hash"] = hash_passcode(phrase)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    return CONFIG_PATH


def _set_pin():
    """A typed PIN for the phone keypad. Safer than a spoken phrase:
    nothing to overhear on a room mic or a stream VOD."""
    import getpass
    from backtalk.config import CONFIG_PATH
    print(f"Writing to: {_config_arg() or CONFIG_PATH}")
    print("Choose a PIN for the phone keypad: 4 to 10 digits.")
    a = getpass.getpass("PIN: ")
    b = getpass.getpass("Again: ")
    a, b = a.strip(), b.strip()
    if a != b:
        print("Not set: the two didn't match.")
        return 1
    if not (a.isdigit() and 4 <= len(a) <= 10):
        print("Not set: digits only, 4 to 10 of them.")
        return 1
    print(f"PIN set in {_write_hash(a)}. Restart the voice line to use it.")
    return 0


if __name__ == "__main__":
    if "--set-pin" in sys.argv:
        sys.exit(_set_pin())
    if "--set-passcode" in sys.argv:
        sys.exit(_set_passcode())
    print("usage: python -m backtalk.remote --set-pin | --set-passcode"
          "\n              [--config <path to backtalk.json>]")
