"""Music mode: notice when the mic is carrying music, not a person.

Built for a routed input that carries ONE thing at a time: music while
it plays, the owner's voice once it stops. Speech comes in bursts with
gaps every second or two; music is a wall of sound. So the detector
counts gaps, it doesn't try to recognize anything:

  IN   ENTER_S of sound with no speech-sized gap (60 s by default), or
       5 s when music mode only just ended on a quiet patch (the gap
       between two playlist tracks shouldn't cost another minute).
  OUT  EXIT_QUIET_S of quiet (the music stopped), or EXIT_GAPS gaps
       inside EXIT_WINDOW_S (someone started talking). The second kind
       is flagged as "speech" so the ears can replay the last few
       seconds and keep the start of the sentence.

A "gap" is a stretch of frames clearly quieter than the last 3 s of
sound, so it needs no calibration for the routing's levels.

While on, it also publishes a 32-band spectrum of what it hears for the
face's visualizer, drawn from the SAME audio she hears, never a second
capture that could be pointed at a different device.

Every frame passes through feed(); keep it cheap. Never raises.
"""
from collections import deque

import numpy as np

from backtalk import signals
from backtalk.config import CFG
from backtalk.vlog import log

FRAME_S = 0.030
_BANDS = 32


class MusicDetector:
    def __init__(self, rate: int):
        self.rate = rate
        self.enter_s = float(CFG.get("music_enter_s") or 60)
        self.reenter_s = 5.0
        self.exit_quiet_s = float(CFG.get("music_exit_quiet_s") or 3)
        self.gap_frames = max(1, int(0.35 / FRAME_S))
        self.exit_gaps, self.exit_window = 3, 8.0
        self.on = False
        self.exited_on_speech = False   # consumed by the ears
        self.speech_frames = 0          # how far back the speech began
        self._recent = deque(maxlen=int(3 / FRAME_S))
        self._gaps: deque = deque()
        self._i = 0                     # frame clock
        self._quiet_run = 0
        self.sound_run = 0              # frames of unbroken sound
        self._left_quiet_at = -10**9    # when we last exited on quiet
        self._spec_buf: list = []
        self._peak = 1e-6
        self._pub_every = 4             # spectrum every 4 frames (~8/s)
        self.gain = float(CFG.get("music_gain") or 0.6)

    @property
    def sound_run_s(self) -> float:
        return self.sound_run * FRAME_S

    def feed(self, frame: np.ndarray, level_db: float):
        """One 30 ms frame plus its level from ears._dbfs (positive dB
        below full scale: SMALLER is LOUDER)."""
        if not CFG.get("music_mode"):
            return
        try:
            self._feed(frame, level_db)
        except Exception as e:
            log(f"[music] detector error (ignored): {e}")

    def _feed(self, frame, level):
        self._i += 1
        med = float(np.median(self._recent)) if len(self._recent) >= 30 else level
        quiet = level > 70 or level > med + 15
        self._recent.append(level)
        if quiet:
            self._quiet_run += 1
            if self._quiet_run == self.gap_frames:
                self._gaps.append(self._i)
                self.sound_run = 0
        else:
            self._quiet_run = 0
            self.sound_run += 1
        horizon = self._i - int(self.exit_window / FRAME_S)
        while self._gaps and self._gaps[0] < horizon:
            self._gaps.popleft()

        if not self.on:
            quick = (self._i - self._left_quiet_at) * FRAME_S < 30
            need = self.reenter_s if quick else self.enter_s
            if self.sound_run_s >= need:
                self._set(True, f"{self.sound_run_s:.0f}s of unbroken sound")
        else:
            if self._quiet_run * FRAME_S >= self.exit_quiet_s:
                self._left_quiet_at = self._i
                self._set(False, "the music stopped")
            elif len(self._gaps) >= self.exit_gaps:
                # the first gap is where the music gave way: replay from
                # just before it (gap length plus a little pre-roll)
                self.speech_frames = (self._i - self._gaps[0]
                                      + self.gap_frames + 10)
                self.exited_on_speech = True
                self._set(False, "speech-shaped gaps")
            else:
                self._spectrum(frame)

    def force_on(self):
        """The spoken "music mode on": skip the wait. The normal exits
        still apply, since nothing is transcribed while it's on."""
        if CFG.get("music_mode") and not self.on:
            self.sound_run = int(self.enter_s / FRAME_S)
            self._set(True, "asked for by voice")

    def force_off(self):
        if self.on:
            self._set(False, "asked for by voice")

    def _set(self, on: bool, why: str):
        self.on = on
        self._gaps.clear()
        self._spec_buf = []
        log(f"[music] music mode {'ON' if on else 'off'} ({why})")
        signals.set_music(on)

    def _spectrum(self, frame):
        self._spec_buf.append(frame)
        if len(self._spec_buf) < self._pub_every:
            return
        x = np.concatenate(self._spec_buf).astype(np.float32) / 32768.0
        self._spec_buf = []
        mag = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        freqs = np.fft.rfftfreq(len(x), 1 / self.rate)
        edges = np.geomspace(40, self.rate / 2 * 0.95, _BANDS + 1)
        bands = np.array([mag[(freqs >= lo) & (freqs < hi)].mean()
                          if np.any((freqs >= lo) & (freqs < hi)) else 0.0
                          for lo, hi in zip(edges[:-1], edges[1:])])
        bands = np.log1p(bands * 50)
        # a slowly decaying peak keeps the bars full-height at any volume
        self._peak = max(float(bands.max()), self._peak * 0.995, 1e-6)
        # The peak normalisation above deliberately fills the bars at any
        # volume, which on a loud source pins every band near the top.
        # music_gain scales the normalised bands back down afterwards so
        # the visualizer has somewhere to go: 1.0 is the old behaviour.
        out = np.clip(bands / self._peak * self.gain, 0, 1)
        signals.set_music(True, out.round(3).tolist())
