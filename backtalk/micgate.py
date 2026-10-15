"""The deadman's switch on the open microphone's gate.

The open-mic loop throws away every frame while a gate is held shut --
the button is down, a phone holds the floor, or the mouth is talking and
barge-in is off. That is correct, right up until the gate never opens
again.

It can stick. `Mouth._speaking` is cleared in a `finally`, so a
`_play_stream` that blocks for ever -- an output device pulled out from
under it mid-sentence is the way this actually happens -- never reaches
that line. The flag stays set, the capture loop keeps running, keeps
discarding, publishes no level and logs no error. The end-user symptom is
a voice line that looks perfectly idle and cannot hear a word, which cost
a whole morning on 2026-10-15.

So: no gate may be trusted indefinitely. Past the limit this forces the
mic open and says so, once.

THE COST, NAMED: if a genuinely long reply is still playing when the
limit expires, the mic hears the tail of our own voice and may answer
it. That is why the limit is well past any single spoken sentence. A
mic that occasionally hears itself is recoverable; one that is deaf for
ever is not.
"""
import time

DEFAULT_LIMIT_S = 25.0


class GateWatchdog:
    """Wraps the open mic's gate. `update(shut)` returns the gate to
    actually obey: it passes the real answer through until the gate has
    been shut continuously for longer than the limit, and then reports
    open regardless, once per stuck stretch."""

    def __init__(self, limit_s: float = DEFAULT_LIMIT_S, clock=time.monotonic):
        self.limit_s = float(limit_s)
        self._clock = clock
        self._shut_since = None
        self._tripped = False

    def update(self, shut: bool) -> bool:
        if not shut:
            # Any honest opening resets everything, including the trip:
            # the next stuck stretch gets its own full allowance and its
            # own log line.
            self._shut_since = None
            self._tripped = False
            return False
        now = self._clock()
        if self._shut_since is None:
            self._shut_since = now
            return True
        if now - self._shut_since >= self.limit_s:
            self._tripped = True
            return False
        return True

    @property
    def tripped(self) -> bool:
        """True while the gate is being overridden -- the caller logs on
        the rising edge so a stuck gate says so exactly once."""
        return self._tripped
