"""The capture stream, reopened when it dies under us.

`_open_mic` in ears.py carries every piece of device recovery backtalk
owns -- a named device that vanished, a sample rate refused, PortAudio's
cached device list gone stale. All of it runs ONCE, at the moment
listening starts.

That left the other half uncovered. `listen_once` then sits in
`stream.read()` indefinitely, and a stream can die after it is open: the
device gets reclaimed by another process, a Bluetooth headset flips
mode, or the audio callback is starved badly enough that PortAudio tears
the stream down. The read raises, the exception escapes the listen loop,
and the voice line is deaf until the whole stack is restarted -- with
nothing logged, because nobody was watching this end.

That is the 2026-10-14 bug: a new and much heavier face went up on the
same machine, the mic started dying at random that night, and it cost a
full day. The gate watchdog added the same day did not touch it, because
the gate was never what was stuck.

So no OPEN stream is trusted for ever either. This wraps one: read
normally, and on any failure close the corpse, open a fresh stream
through the same recovery path, and retry. It says so once per outage --
paced, so a device that is away for a minute neither spins the CPU nor
floods the log.

THE COST, NAMED: a frame or two of audio is lost at the moment of the
reopen, so a word spoken exactly as the device dies may be clipped. A
clipped word is recoverable; a microphone that is deaf until reboot is
not.
"""
import time

DEFAULT_RETRY_GAP_S = 0.25
# How many reopen attempts may pass before we stop talking about this
# outage. Three lines is enough to find it in a log; thirty is noise.
_MAX_LINES_PER_OUTAGE = 3


class ReopeningStream:
    """A capture stream that replaces itself when it dies.

    `open_stream()` must return something with `.read(n) -> (block, flag)`
    and `.close()`; in production that is `ears._open_mic`, so every
    device fallback it already implements is inherited rather than
    duplicated here.
    """

    def __init__(self, open_stream, log=print, clock=time.monotonic,
                 sleep=time.sleep, retry_gap_s: float = DEFAULT_RETRY_GAP_S):
        self._open = open_stream
        self._log = log
        self._clock = clock
        self._sleep = sleep
        self.retry_gap_s = float(retry_gap_s)
        self._stream = None

    # -- context manager -----------------------------------------------
    def __enter__(self):
        if self._stream is None:
            self._stream = self._open()
        return self

    def __exit__(self, *exc):
        self._close()
        return False

    def _close(self):
        s, self._stream = self._stream, None
        if s is None:
            return
        try:
            s.close()
        except Exception:
            pass        # a dead stream often refuses to close; let it go

    # -- the read ------------------------------------------------------
    def read(self, frames):
        """Read one block, reopening the device as many times as it
        takes. Only ever returns real audio."""
        while True:
            if self._stream is None:
                self._stream = self._open()
            try:
                return self._stream.read(frames)
            except Exception as e:
                self._recover(e)

    def _recover(self, why):
        """The stream just died. Bury it and open another."""
        self._close()
        said = 0
        self._log(f"[ears] the microphone stream died ({why}) -- reopening")
        said += 1
        attempts = 0
        while True:
            attempts += 1
            try:
                self._stream = self._open()
            except Exception as e:
                if said < _MAX_LINES_PER_OUTAGE:
                    said += 1
                    self._log(f"[ears] could not reopen the microphone "
                              f"({e}) -- still trying")
                # Paced, not spun: the device is usually back within a
                # second or two, and a tight loop here would make the
                # starvation that caused this worse.
                self._sleep(self.retry_gap_s)
                continue
            if attempts > 1:
                self._log("[ears] the microphone is back")
            return
