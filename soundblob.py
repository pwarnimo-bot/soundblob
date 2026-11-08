#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Pol Warnimont
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; see the LICENSE file for details.
r"""
soundblob — a colorful, weird, googly-eyed blob that bounces around your
desktop to whatever music is playing.

* Captures system audio from the default sink monitor (PulseAudio/PipeWire via parec)
* FFT + beat detection -> jumps, squish, spectrum-shaped wobble, particle bursts
* Transparent, click-through Wayland layer-shell overlay (sway/wlroots)

Quit with Ctrl+C in the terminal, or: pkill -f '^python3 \S*soundblob\.py( |$)'
Toggle the FPS readout: pkill -USR1 -f '^python3 \S*soundblob\.py( |$)'
Toggle the oscilloscope: pkill -USR2 -f '^python3 \S*soundblob\.py( |$)'
"""
import argparse
import collections
import colorsys
import math
import random
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import cairo
import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gtk, Gdk, GLib, Gio  # noqa: E402

try:
    gi.require_version("GtkLayerShell", "0.1")
    from gi.repository import GtkLayerShell  # noqa: E402
except (ValueError, ImportError):
    GtkLayerShell = None

RATE = 44100
CHUNK = 512          # samples per read (~11.6 ms)
FFT_SIZE = 2048
TAU = math.tau
FPS = RATE / CHUNK   # analysis frames per second (~86)

# Confetti budget. Every particle costs ~0.02 ms of CPU drawing per frame, so keep it modest.
# --confetti scales both (0 = none, 1 = default, 2 = double).
CONFETTI_RATE = 0.35      # fraction of requested particles that actually spawn
MAX_PARTICLES = 120       # hard cap on live particles
SCOPE_SAMPLES = 1024      # oscilloscope window (~23 ms of audio)
SCOPE_PIECES = 16         # rainbow colour segments along the scope line
TRAIL_MAX = 18            # ghost-trail circles at full craziness (each is a big translucent fill)

# Mood calibration, measured on real rock/metal recordings and synthetic techno/psytrance.
# Spectral flatness (200 Hz-5 kHz): clean/acoustic guitars 0.27-0.52, distorted metal 0.67-0.73
FLAT_CLEAN, FLAT_DIRTY = 0.50, 0.68
W_LOUD = 0.25      # loudness vs. the current song's loud parts (quiet passages -> lazy)
W_AGGR = 0.30      # distortion (rock/metal guitars)
W_TEMPO = 0.15     # fast, *reliable* tempo
W_DENSITY = 0.20   # drum hits per second
W_DRIVE = 0.30     # rock-steady fast grid: full from 136 BPM (psytrance), half at 130 (techno)
W_METAL = 0.30     # distortion + fast kicks: thrash / double bass / blast beats
KICK_LOWNESS, SNARE_LOWNESS = 8.0, 6.0
# loudness reference (seconds): rise, fall, and silence that marks a song change
LOUD_RISE, LOUD_FALL, LOUD_RESET_SILENCE = 5.0, 40.0, 1.0
# "heavy" (dark body + pentagram): distortion above the horns' range, at the song's loud parts
HEAVY_AGGR_LO, HEAVY_AGGR_HI = 0.55, 0.85
DARK = (0.07, 0.0, 0.02)   # the colour a fully heavy blob fades to
VIGNETTE_ALPHA = 0.65      # darkness of the screen corners at full heaviness
# "mellow" (smile, rosy cheeks, gentle sway): clean guitars at low energy (acoustic, ballads,
# quiet grunge). Fully mellow below MELLOW_E_LO energy, gone above MELLOW_E_HI or with distortion.
MELLOW_E_LO, MELLOW_E_HI = 0.28, 0.48
MELLOW_AGGR = 0.35


# --------------------------------------------------------------------------- audio
def clip01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


class Onsets:
    """Adaptive peak picker for one onset-strength signal (median + k*MAD)."""

    def __init__(self, k, refractory, floor, rel=0.0, hold=6.0):
        self.hist = collections.deque(maxlen=int(FPS * 1.5))
        self.k, self.refractory, self.floor = k, refractory, floor
        # must also reach `rel` x a slowly decaying peak-hold (half-life `hold` s),
        # so quiet stuff can't sneak through right after the drums stop
        self.rel = rel
        self.pk = 0.0
        self.pk_decay = 0.5 ** (1 / (hold * FPS))
        self.freeze = False        # hold the peak (e.g. during a breakdown)
        self.last = -1e9
        self.times = collections.deque()
        self.count = 0
        self.strength = 0.0
        self.thr = 0.0

    def feed(self, val, now, gate=True):
        self.hist.append(val)
        self.pk = max(val, self.pk if self.freeze else self.pk * self.pk_decay)
        while self.times and now - self.times[0] > 3.0:
            self.times.popleft()
        if len(self.hist) < 30:
            return False
        arr = np.fromiter(self.hist, dtype=np.float32)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        self.thr = med + self.k * 1.4826 * mad + self.floor
        if (gate and val > self.thr and val >= self.rel * self.pk
                and now - self.last > self.refractory):
            self.last = now
            self.strength = min(val / (self.thr + 1e-6), 4.0)
            self.count += 1
            self.times.append(now)
            return True
        return False

    def ratio(self, val):
        return val / (self.thr + 1e-6)

    def reset(self):
        """New song: forget the previous song's levels."""
        self.hist.clear()
        self.pk = 0.0
        self.freeze = False

    @property
    def rate(self):
        return len(self.times) / 3.0


class Audio(threading.Thread):
    """Reads the sink monitor and publishes band energies, drum hits, tempo,
    song-structure events (breakdowns / drops) and an overall 'energy' mood."""

    def __init__(self, device):
        super().__init__(daemon=True)
        self.device = device
        self.window = np.hanning(FFT_SIZE).astype(np.float32)
        self.buf = np.zeros(FFT_SIZE, dtype=np.float32)
        freqs = np.fft.rfftfreq(FFT_SIZE, 1 / RATE)

        def band(lo, hi):
            return np.where((freqs >= lo) & (freqs < hi))[0]

        self.band_bass = band(30, 160)
        self.band_mid = band(160, 2000)
        self.band_high = band(2000, 12000)
        self.b_kick = band(35, 120)          # kick drum fundamental
        self.b_snare_lo = band(150, 320)     # snare body
        self.b_snare_hi = band(1500, 6000)   # snare crack
        self.b_hat = band(7000, 16000)       # hi-hats / cymbals / risers
        self.b_body = band(200, 5000)        # where distorted guitars live
        self.b_all = band(30, 16000)
        # 32 log-spaced bins for the blob's outline
        edges = np.geomspace(40, 14000, 33)
        self.spec_idx = [np.where((freqs >= edges[i]) & (freqs < edges[i + 1]))[0]
                         for i in range(32)]
        self.prev_log = None
        self.prev_lin = None
        self.peak = 1e-4           # auto gain
        self.frame = 0

        # drum detectors (refractory: kicks allow double-bass 16ths, ~13/s)
        self.kick = Onsets(k=2.2, refractory=0.07, floor=0.018, rel=0.5, hold=6.0)
        self.snare = Onsets(k=2.6, refractory=0.09, floor=0.0, rel=0.3, hold=4.0)
        self.hat = Onsets(k=2.6, refractory=0.06, floor=0.05)
        self.onset = Onsets(k=2.0, refractory=0.10, floor=0.04)   # anything

        # tempo
        self.env = collections.deque(maxlen=int(FPS * 6))
        self.bpm = 0.0
        self.tempo_conf = 0.0
        self._bpm_cand = 0.0
        self._bpm_votes = 0

        # published state (read by UI thread)
        self.level = 0.0
        self.bass = 0.0
        self.mid = 0.0
        self.high = 0.0
        self.spectrum = np.zeros(32, dtype=np.float32)
        # event counters; the UI watches for increments
        self.beats = 0             # generic onsets (fallback when there are no drums)
        self.beat_strength = 0.0
        self.kicks = 0
        self.kick_strength = 0.0
        self.snares = 0
        self.snare_strength = 0.0
        self.drops = 0
        self.drop_strength = 0.0
        # song structure
        self.breakdown = False     # music playing but the kick is gone
        self.groove_break = False  # ...and there WAS a groove before (a real breakdown, not an intro)
        self._kick_long = collections.deque()
        self.breakdown_len = 0.0
        self.build = 0.0           # riser / snare-roll intensity during a breakdown
        self._bd_start = 0.0
        self._bd_high = 0.0
        self._active_since = None
        self.kick_rate = 0.0
        self.snare_rate = 0.0
        self.onset_rate = 0.0
        self.flatness = 0.0
        self.aggression = 0.0      # distorted, dense, loud -> metal
        # mood
        self.short_rms = 0.0
        self.ref_rms = 0.0         # "how loud is loud" in the current song
        self._silent_frames = 0
        self.loud_rel = 1.0
        self.energy = 0.4          # 0 = lazy, 0.5 = normal, 1 = crazy
        self.mellow = 0.0          # calm, clean music playing: the blob smiles and sways
        self.parts = {}            # weighted mood ingredients (for --debug)
        self.running = True

    def run(self):
        while self.running:
            try:
                proc = subprocess.Popen(
                    ["parec", f"--device={self.device}", "--format=float32le",
                     f"--rate={RATE}", "--channels=1", "--latency-msec=15", "--raw"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                print("parec not found — install libpulse / pipewire-pulse", file=sys.stderr)
                return
            nbytes = CHUNK * 4
            while self.running:
                data = proc.stdout.read(nbytes)
                if not data or len(data) < nbytes:
                    break
                self.process(np.frombuffer(data, dtype=np.float32))
            proc.kill()
            time.sleep(0.5)  # restart if parec died (e.g. sink changed)

    def process(self, samples):
        now = time.monotonic()
        self.frame += 1
        # build the new buffer fully before publishing it: the UI thread reads it for the scope
        buf = np.roll(self.buf, -len(samples))
        buf[-len(samples):] = samples
        self.buf = buf
        mag = np.abs(np.fft.rfft(self.buf * self.window))

        rms = float(np.sqrt(np.mean(samples ** 2)))
        self.peak = max(self.peak * 0.9995, rms, 1e-4)
        norm = 1.0 / (self.peak * 40 + 1e-6)

        bass = float(mag[self.band_bass].mean()) * norm
        mid = float(mag[self.band_mid].mean()) * norm * 3
        high = float(mag[self.band_high].mean()) * norm * 10
        spec = np.array([mag[i].mean() if len(i) else 0 for i in self.spec_idx], dtype=np.float32)
        spec = np.log1p(spec * norm * 4)
        spec /= max(spec.max(), 1e-3)

        a = 0.5
        self.bass = self.bass * a + min(bass, 3) * (1 - a)
        self.mid = self.mid * a + min(mid, 3) * (1 - a)
        self.high = self.high * a + min(high, 3) * (1 - a)
        self.level = min(rms / self.peak, 1.5) if rms > 1e-4 else 0.0
        self.spectrum = self.spectrum * 0.6 + spec * 0.4

        # ---- drum detection on spectral flux per band
        lin = mag * norm
        logm = np.log1p(lin * 20)
        playing = rms > 0.002
        if self.prev_log is not None:
            pos = np.maximum(logm - self.prev_log, 0)        # log flux: tempo & generic onsets
            posl = np.maximum(lin - self.prev_lin, 0)        # linear flux: drums
            f_kick = float(posl[self.b_kick].mean())
            f_crack = float(posl[self.b_snare_hi].mean())
            f_snare = math.sqrt(float(posl[self.b_snare_lo].mean()) * f_crack)  # body AND crack
            # kicks are low-heavy; snares/claps/rolls have lots of crack.
            # (measured: real kicks >8, synthetic snare rolls 6-9, hats/snares far lower)
            lowness = f_kick / (f_crack + 1e-6)
            if self.kick.feed(f_kick, now, gate=playing and lowness >= KICK_LOWNESS):
                self.kicks += 1
                self.kick_strength = self.kick.strength
                self.on_kick(now)
            if self.snare.feed(f_snare, now, gate=playing and lowness < SNARE_LOWNESS):
                self.snares += 1
                self.snare_strength = self.snare.strength
            self.hat.feed(float(pos[self.b_hat].mean()), now, gate=playing)
            if self.onset.feed(float(pos[self.b_all].mean()), now, gate=playing):
                self.beats += 1
                self.beat_strength = self.onset.strength
            self.env.append(float(pos[self.b_kick].mean()) +
                            0.6 * float(pos[self.b_snare_hi].mean()) +
                            0.2 * float(pos[self.b_hat].mean()))
        self.prev_log = logm
        self.prev_lin = lin

        if self.frame % 43 == 0:      # ~2x per second
            self.estimate_tempo()

        # ---- distortion: spectral flatness where guitars live
        body = mag[self.b_body] + 1e-9
        flat = float(np.exp(np.mean(np.log(body))) / np.mean(body))
        self.flatness += (flat - self.flatness) * 0.02

        self.update_structure(now, playing)
        self.update_energy(rms)

    def on_kick(self, now):
        self._kick_long.append(now)
        if self.breakdown and self.breakdown_len > 4.0:
            # the kick is back after a breakdown: THE DROP
            self.drop_strength = 1.0 + self.build + min(self.breakdown_len / 16, 1.0)
            self.drops += 1
        self.breakdown = False
        self.groove_break = False
        self.build = 0.0
        self.kick.freeze = False

    def estimate_tempo(self):
        if len(self.env) < FPS * 3:
            return
        e = np.fromiter(self.env, dtype=np.float32)
        e -= e.mean()
        n = len(e)
        spec = np.fft.rfft(e, 2 * n)
        ac = np.fft.irfft(spec * np.conj(spec))[:n]
        if ac[0] <= 0:
            return
        lo, hi = int(FPS * 60 / 210), int(FPS * 60 / 60)
        lags = np.arange(lo, hi + 1)
        bpms = 60 * FPS / lags
        prior = np.exp(-0.5 * (np.log2(bpms / 125) / 0.9) ** 2)
        score = ac[lags] / ac[0] * prior
        i = int(np.argmax(score))
        # parabolic refinement of the lag
        lag = float(lags[i])
        if 0 < i < len(lags) - 1:
            y0, y1, y2 = score[i - 1], score[i], score[i + 1]
            d = y0 - 2 * y1 + y2
            if d != 0:
                lag += 0.5 * (y0 - y2) / d
        raw = 60 * FPS / lag
        self.tempo_conf = float(ac[lags[i]] / ac[0])
        if self.bpm == 0 or abs(raw - self.bpm) / self.bpm < 0.06:
            self.bpm = raw if self.bpm == 0 else self.bpm * 0.7 + raw * 0.3
            self._bpm_votes = 0
        else:
            # only jump to a new tempo if it's consistent for a few estimates
            if self._bpm_cand and abs(raw - self._bpm_cand) / self._bpm_cand < 0.06:
                self._bpm_votes += 1
            else:
                self._bpm_cand, self._bpm_votes = raw, 1
            if self._bpm_votes >= 3:
                self.bpm, self._bpm_votes = raw, 0

    @property
    def beat_period(self):
        """Seconds per beat, or 0 if tempo is unknown/unreliable."""
        return 60.0 / self.bpm if self.bpm and self.tempo_conf > 0.3 else 0.0

    def update_structure(self, now, playing):
        self.kick_rate = self.kick.rate
        self.snare_rate = self.snare.rate
        self.onset_rate = self.onset.rate
        active = self.short_rms > 0.004
        if not active:
            self._active_since = None
        elif self._active_since is None:
            self._active_since = now
        since_kick = now - max(self.kick.last, self._active_since or now)
        if active and since_kick > 2.5:
            if not self.breakdown:
                self.breakdown = True
                self._bd_start = now - since_kick
                while self._kick_long and self._bd_start - self._kick_long[0] > 20:
                    self._kick_long.popleft()
                self.groove_break = len(self._kick_long) >= 8
                self._bd_high = max(self.high, 0.05)
            self.breakdown_len = now - self._bd_start
            # remember how hard the kick hit before the breakdown: only a kick
            # of similar power counts as the drop (not a riser or snare roll)
            self.kick.freeze = self.breakdown_len < 24
            # build-up: rising highs (risers / white-noise sweeps) + snare rolls
            rise = clip01((self.high / self._bd_high - 1.0) / 1.5)
            roll = clip01((self.snare_rate + self.hat.rate * 0.5 - 2) / 8)
            target = 0.6 * rise + 0.4 * roll
            self.build += (target - self.build) * 0.02
        elif not active:
            self.breakdown = False
            self.groove_break = False
            self.build = 0.0
            self.kick.freeze = False

    def update_energy(self, rms):
        """How hyped is the music right now? Loudness within the current song,
        drum density, tempo and distortion."""
        self.short_rms += (rms - self.short_rms) * 0.06          # ~0.2 s
        # Loudness only counts *within* a song: between songs it mostly reflects
        # mastering, streaming volume normalisation and the volume knob, not energy.
        # So the reference resets at a song change (>1 s of silence).
        if self.short_rms < 0.003:
            self._silent_frames += 1
            if int(self._silent_frames) == int(FPS * LOUD_RESET_SILENCE):
                self.song_change()
        else:
            self._silent_frames = 0
            if self.ref_rms == 0.0:
                self.ref_rms = self.short_rms
            # rises within ~5 s, settles to a new level over ~40 s, so quieter
            # passages and breakdowns read as quieter than the song's loud parts
            k = 1 / (LOUD_RISE * FPS) if self.short_rms > self.ref_rms else 1 / (LOUD_FALL * FPS)
            self.ref_rms += (self.short_rms - self.ref_rms) * k
        self.loud_rel = self.short_rms / max(self.ref_rms, 1e-4)

        loud = clip01((self.loud_rel - 0.35) / 0.75)
        aggr_target = (clip01((self.flatness - FLAT_CLEAN) / (FLAT_DIRTY - FLAT_CLEAN)) *
                       clip01(self.high) * clip01(self.loud_rel * 1.5))
        self.aggression += (aggr_target - self.aggression) * 0.02
        steady = clip01((self.tempo_conf - 0.3) / 0.3) if self.bpm else 0.0
        tempo = clip01((self.bpm - 100) / 45) * steady
        drive = clip01((self.bpm - 124) / 12) * steady
        density = clip01((self.kick_rate + self.snare_rate + 0.5 * self.hat.rate +
                          0.3 * self.onset_rate - 2.5) / 5)
        metal = self.aggression * clip01((self.kick_rate - 1.5) / 2)
        self.parts = dict(loud=W_LOUD * loud, aggr=W_AGGR * self.aggression, tempo=W_TEMPO * tempo,
                          density=W_DENSITY * density, drive=W_DRIVE * drive, metal=W_METAL * metal)
        raw = clip01(sum(self.parts.values()))
        if self.breakdown:
            raw *= 0.75 + 0.5 * self.build      # tension rises with the build-up
        if self.short_rms < 0.003:
            raw = 0.0
        self.energy += (raw - self.energy) * (0.03 if raw > self.energy else 0.01)

        # mellow: calm, clean (undistorted) music that is actually playing,
        # not a techno breakdown waiting for its drop
        calm = clip01((MELLOW_E_HI - self.energy) / (MELLOW_E_HI - MELLOW_E_LO))
        clean = 1 - clip01(self.aggression / MELLOW_AGGR)
        target = calm * clean
        if self.short_rms < 0.003 or self.groove_break:
            target = 0.0
        self.mellow += (target - self.mellow) * 0.005           # ~2.5 s

    def song_change(self):
        """>1 s of silence: a new song starts. Every adaptive level (auto gain,
        drum thresholds, loudness reference, tempo) starts over, so a quieter
        song after a loud one isn't judged against the loud one."""
        self.ref_rms = 0.0
        self.peak = 1e-4
        for det in (self.kick, self.snare, self.hat, self.onset):
            det.reset()
        self.env.clear()
        self.bpm = 0.0
        self.tempo_conf = 0.0
        self._bpm_votes = 0
        self._kick_long.clear()

    def debug_line(self):
        return (f"bpm={self.bpm:5.1f} conf={self.tempo_conf:.2f} kick/s={self.kick_rate:4.1f} "
                f"snare/s={self.snare_rate:4.1f} onset/s={self.onset_rate:4.1f} "
                f"flat={self.flatness:.3f} aggr={self.aggression:.2f} loud={self.loud_rel:.2f} "
                f"bd={'Y' if self.groove_break else ('i' if self.breakdown else '-')} build={self.build:.2f} "
                f"drops={self.drops} energy={self.energy:.2f} mellow={self.mellow:.2f} ["
                + " ".join(f"{k}={v:.2f}" for k, v in self.parts.items()) + "]")


# --------------------------------------------------------------------------- now playing
# Mark Lanegan easter egg. Any track with "lanegan" among Spotify's artists (his solo work,
# Mark Lanegan Band, duets and credited features) or in the title ("feat. Mark Lanegan")
# counts. These cover the songs he sings on without being listed as an artist.
# Names are compared lowercase, without punctuation or " - Remastered"/"(Live)" suffixes.
LANEGAN_BANDS = {"screaming trees", "the gutter twins"}     # every song: he was the singer
LANEGAN_ALBUMS = {                                           # (artist, album): all of it
    ("soulsavers", "its not how far you fall its the way you land"),
    ("soulsavers", "broken"),
}
LANEGAN_SONGS = {                                            # (artist, title)
    ("queens of the stone age", "in the fade"),
    ("queens of the stone age", "hangin tree"),
    ("queens of the stone age", "a song for the dead"),
    ("queens of the stone age", "god is in the radio"),
    ("queens of the stone age", "this lullaby"),
    ("queens of the stone age", "burn the witch"),
    ("mad season", "long gone day"),
    ("mad season", "im above"),
    ("mad season", "locomotive"),
    ("mad season", "black book of fear"),
    ("mad season", "slip away"),
}


def _norm(name):
    """'Hangin' Tree - 2002 Remaster' -> 'hangin tree'"""
    name = name.lower().split(" - ")[0].split(" (")[0].split(" [")[0]
    return " ".join("".join(c for c in name if c.isalnum() or c.isspace()).split())


def lanegan_on(artists, title, album=""):
    """Is Mark Lanegan singing on this track?"""
    if any("lanegan" in a.lower() for a in artists) or "lanegan" in title.lower():
        return True
    names = {_norm(a) for a in artists}
    t, al = _norm(title), _norm(album)
    return bool(names & LANEGAN_BANDS or any((n, al) in LANEGAN_ALBUMS or (n, t) in LANEGAN_SONGS
                                              for n in names))


class NowPlaying:
    """Asks Spotify (over MPRIS on the session bus) what's playing, every couple of seconds.
    Easter egg: Mark Lanegan gets the blob into sunglasses and a cigarette."""

    BUS_NAME = "org.mpris.MediaPlayer2.spotify"

    def __init__(self):
        self.artists = []
        self.title = ""
        self.album = ""
        self.playing = False
        try:
            self.bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        except GLib.Error:
            self.bus = None
            return
        self.poll()
        GLib.timeout_add_seconds(2, self.poll)

    def poll(self):
        # asynchronous, so a hung or missing Spotify never stalls the animation
        for prop, done in (("Metadata", self._on_metadata), ("PlaybackStatus", self._on_status)):
            self.bus.call(self.BUS_NAME, "/org/mpris/MediaPlayer2", "org.freedesktop.DBus.Properties",
                          "Get", GLib.Variant("(ss)", ("org.mpris.MediaPlayer2.Player", prop)),
                          GLib.VariantType("(v)"), Gio.DBusCallFlags.NO_AUTO_START, 1000, None, done)
        return GLib.SOURCE_CONTINUE

    def _get(self, conn, res):
        try:
            return conn.call_finish(res).unpack()[0]
        except GLib.Error:           # Spotify isn't running
            return None

    def _on_metadata(self, conn, res):
        md = self._get(conn, res) or {}
        self.artists = list(md.get("xesam:artist") or [])
        self.title = md.get("xesam:title") or ""
        self.album = md.get("xesam:album") or ""

    def _on_status(self, conn, res):
        self.playing = self._get(conn, res) == "Playing"

    @property
    def lanegan(self):
        return self.playing and lanegan_on(self.artists, self.title, self.album)


# --------------------------------------------------------------------------- helpers
def hsv(h, s=1.0, v=1.0):
    return colorsys.hsv_to_rgb(h % 1.0, s, v)


def smooth_closed_path(cr, pts):
    """Catmull-Rom spline through pts, as cairo beziers."""
    n = len(pts)
    cr.move_to(*pts[0])
    for i in range(n):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        cr.curve_to(*c1, *c2, *p2)
    cr.close_path()


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max", "hue", "size", "kind", "rot")

    def __init__(self, x, y, vx, vy, life, hue, size, kind):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.life = self.max = life
        self.hue, self.size, self.kind = hue, size, kind
        self.rot = random.uniform(0, TAU)


# --------------------------------------------------------------------------- the blob
def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def lerp(a, b, t):
    return a + (b - a) * t


def quiet_level(level):
    return level < 0.02


class Blob:
    N = 72  # outline points

    def __init__(self, base_radius, confetti=1.0):
        self.R = base_radius
        self.confetti_rate = min(1.0, CONFETTI_RATE * confetti)
        self.max_particles = int(MAX_PARTICLES * confetti)
        self.x, self.y = 300.0, 200.0
        self.vx, self.vy = 220.0, 0.0
        self.hue = random.random()
        self.t = 0.0
        self.anim_t = 0.0          # animation clock: slow when lazy, fast when crazy
        self.mood = 0.4            # 0 lazy .. 0.5 normal .. 1 crazy (smoothed)
        self.g = 1700.0
        self.squash = 0.0          # spring-driven deformation
        self.squash_v = 0.0
        self.squash_ang = 0.0
        self.spin = 0.0
        self.pupils = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]  # x,y,vx,vy (lagging)
        self.blink = 0.0
        self.yawn = 0.0            # 1 -> 0 while yawning
        self.shake = (0.0, 0.0)
        self.on_floor = False
        self.trail = []
        self.particles = []
        self.seen_beats = 0
        self.sleepiness = 0.0
        self.eye_sizes = (1.0, 0.75)
        self.third_eye = 0.0       # pops open on big drops
        # genre flavours
        self.seen_kicks = 0
        self.seen_snares = 0
        self.seen_hats = 0
        self.seen_drops = 0
        self.nod = 0.0             # headbang on the snare (rock/metal)
        self.nod_side = 1
        self.horns = 0.0           # grow with distorted guitars (metal)
        self.heavy = 0.0           # really heavy riffs: body goes dark, pentagram on the forehead
        self.penta_pulse = 0.0     # pentagram flares on kicks and snares
        # bloodshot veins per eye: (angle, length, kink) — fixed so they don't flicker
        self.veins = [[(random.uniform(0, TAU), random.uniform(0.35, 0.6), random.uniform(-0.5, 0.5))
                       for _ in range(7)] for _ in range(2)]
        self.rumble = 0.0          # double-bass / blast-beat vibration
        self.hover = 0.0           # floats during breakdowns (techno/psy)
        self.spiral = 0.0          # hypnotic eyes during build-ups (psy)
        self.charge = 0.0          # build-up tension
        self.drop_boost = 0.0      # forced craziness after a drop
        self.flash = 0.0           # screen flash on the drop
        self.land_at = 0.0         # when a beat-synced jump will touch down
        self.mellow = 0.0          # smiles, blushes and sways to calm music
        self.sway = 0.0            # sway phase (one side-to-side swing per two beats)
        self.bliss = 0.0           # 1 -> 0 while it closes its eyes, enjoying the song
        self.cool = 0.0            # easter egg: sunglasses + cigarette (Mark Lanegan on Spotify)
        self.cool_target = 0.0
        self.ember = None          # where the cigarette's tip was drawn (smoke rises from there)

    @property
    def lazy(self):
        return clamp01((0.38 - self.mood) / 0.3)

    @property
    def evil(self):
        """How visible the pentagram is: also drives the dark, bloody eyes."""
        return clamp01((self.heavy - 0.15) / 0.45)

    @property
    def crazy(self):
        return clamp01((self.mood - 0.65) / 0.25)

    # ------------------------------------------------------------------ physics
    def update(self, dt, W, H, a):
        self.t += dt
        level, bass, high = a.level, a.bass, a.high

        self.drop_boost = max(0.0, self.drop_boost - dt / 7.0)
        target = max(a.energy, self.drop_boost) if level > 0.02 else 0.0
        self.mood += (target - self.mood) * min(1.0, dt * (1.5 + 6 * self.drop_boost))
        m, L, C = self.mood, self.lazy, self.crazy
        self.anim_t += dt * (0.45 + 1.6 * m)

        # song structure -> breakdown hover, build-up charge, hypnotic eyes
        bd = 1.0 if a.groove_break and not quiet_level(level) else 0.0
        self.hover += (bd * clamp01(0.45 + a.build) - self.hover) * min(1, dt * 1.5)
        self.charge += (bd * a.build - self.charge) * min(1, dt * 3)
        self.spiral += ((1.0 if bd and a.build > 0.3 else 0.0) - self.spiral) * min(1, dt * 2)
        # metal flavours
        self.horns += (clamp01((a.aggression - 0.45) / 0.35) - self.horns) * min(1, dt * 0.8)
        self.rumble += (clamp01((a.kick_rate - 3.5) / 3) * clamp01(a.aggression * 2) * (1 - bd)
                        - self.rumble) * min(1, dt * 3)
        heavy = (clamp01((a.aggression - HEAVY_AGGR_LO) / (HEAVY_AGGR_HI - HEAVY_AGGR_LO)) *
                 clamp01((a.loud_rel - 0.45) / 0.3) * (1 - bd))
        if quiet_level(level):
            heavy = 0.0
        # darkness creeps in over ~1.5 s and lifts more slowly
        self.heavy += (heavy - self.heavy) * min(1, dt * (0.7 if heavy > self.heavy else 0.35))
        self.penta_pulse = max(0.0, self.penta_pulse - dt * 4)
        self.nod = max(0.0, self.nod - dt * 7)
        self.flash = max(0.0, self.flash - dt * 3)
        self.cool += (self.cool_target - self.cool) * min(1, dt * 1.5)
        if self.cool > 0.5 and self.ember and random.random() < dt * 7:
            ex, ey = self.ember
            self.particles.append(Particle(
                ex, ey, random.uniform(-8, 12), random.uniform(-55, -35),
                random.uniform(1.6, 2.6), 0.0, random.uniform(2, 4), "smoke"))
        mel = 0.0 if quiet_level(level) else a.mellow
        self.mellow += (mel - self.mellow) * min(1, dt * 1.2)
        M = self.mellow
        P = a.beat_period
        self.sway += dt * (math.pi / P if P else 1.4)

        self.g = lerp(1100.0, 2500.0, m) * (1 - 0.8 * self.hover)   # lazy/breakdown floats, crazy snaps
        r = self.radius(bass)

        # drum events
        now = time.monotonic()
        if a.drops != self.seen_drops:
            fresh = a.drops - self.seen_drops < 3
            self.seen_drops = a.drops
            if fresh:
                self.on_drop(a.drop_strength, W, H)
        if a.kicks != self.seen_kicks:
            new = a.kicks - self.seen_kicks
            self.seen_kicks = a.kicks
            if new < 5:
                self.on_kick(a.kick_strength, W, H, a)
        if a.snares != self.seen_snares:
            new = a.snares - self.seen_snares
            self.seen_snares = a.snares
            if new < 5:
                self.on_snare(a.snare_strength)
        if a.hat.count != self.seen_hats:
            new = a.hat.count - self.seen_hats
            self.seen_hats = a.hat.count
            if new < 5:
                self.on_hat()
        # generic onsets drive jumps only when there are no drums (intros, ambient, acoustic)
        if a.beats != self.seen_beats:
            new = a.beats - self.seen_beats
            self.seen_beats = a.beats
            if new < 5 and now - a.kick.last > 1.5:
                self.jump(a.beat_strength, W, H, a, synced=False)

        quiet = level < 0.02
        self.sleepiness = min(1, self.sleepiness + dt * 0.4) if quiet else max(0, self.sleepiness - dt * 3)

        self.vy += self.g * dt
        self.vx *= (1 - (0.25 + 1.5 * L) * dt)
        if self.hover > 0.05:
            # levitate around the middle of the screen, bobbing to the music
            target_y = H * 0.42 + math.sin(self.t * 1.3) * H * 0.06 - self.charge * H * 0.08
            self.vy += (target_y - self.y) * 6 * self.hover * dt
            self.vy *= (1 - 2.5 * self.hover * dt)
            self.vx *= (1 - 1.5 * self.hover * dt)
        # build-up / double-bass rumble: vibrating with tension
        jit = self.charge * 7 + self.rumble * 5
        # wobbly mid-air drift with the mids — it wanders
        self.vx += math.sin(self.t * 1.7) * a.mid * 400 * (0.3 + 1.4 * m) * dt
        # mellow: drifts gently from side to side with the music
        self.vx += math.cos(self.sway) * M * 160 * dt

        # crazy: jittery shakes and sudden mid-air direction zaps
        if C > 0 or jit > 0.05:
            self.vx += random.gauss(0, 1) * C * 2500 * dt
            if self.land_at < time.monotonic():   # don't disturb beat-synced arcs
                self.vy += random.gauss(0, 1) * C * 1200 * dt
            amp = C * 5 + jit
            self.shake = (random.gauss(0, 1) * amp, random.gauss(0, 1) * amp)
            if self.rumble > 0.2 and self.on_floor and random.random() < self.rumble * 0.5:
                self.emit(Particle(
                    self.x + random.uniform(-r, r), H - 2, random.uniform(-150, 150), random.uniform(-350, -120),
                    random.uniform(0.2, 0.5), self.hue + 0.5, random.uniform(2, 4), "spark"))
            if C > 0.4 and self.hover < 0.3 and random.random() < dt * C * 1.5:
                self.vx = -self.vx * 1.2 + random.uniform(-300, 300)
                self.squash_v += 3 * C
                self.squash_ang = 0.0
                for _ in range(8):
                    ang = random.uniform(0, TAU)
                    self.emit(Particle(
                        self.x, self.y, math.cos(ang) * 300, math.sin(ang) * 300,
                        0.4, self.hue + random.random(), 4, "star"))
        else:
            self.shake = (0.0, 0.0)

        self.x += self.vx * dt
        self.y += self.vy * dt
        self.spin += self.vx * dt / max(r, 1) * 0.5

        # walls: crazy is bouncy rubber, lazy is a wet sponge
        bounce = lerp(0.45, 0.95, m)
        hit = None
        self.on_floor = False
        if self.x - r < 0:
            self.x = r; hit = (abs(self.vx), 0.0); self.vx = abs(self.vx) * bounce
        elif self.x + r > W:
            self.x = W - r; hit = (abs(self.vx), math.pi); self.vx = -abs(self.vx) * bounce
        if self.y - r < 0:
            self.y = r; hit = (abs(self.vy), math.pi / 2); self.vy = abs(self.vy) * 0.7
        elif self.y + r > H:
            self.y = H - r
            self.on_floor = True
            if self.vy > 0:
                hit = (abs(self.vy), -math.pi / 2)
                self.vy = -abs(self.vy) * lerp(0.35, 0.8, m)
                if abs(self.vy) < 60:
                    self.vy = 0
            self.vx *= (1 - 2.0 * dt)  # floor friction
        if hit and hit[0] > 250:
            self.squash_v += min(hit[0] / 1800, 0.9) * 6
            self.squash_ang = hit[1]
            self.splash(hit[0], hit[1], r)

        # squash spring; a lazy blob sags into a puddle when resting on the floor
        rest = 0.0
        if self.on_floor and self.vy == 0:
            rest = 0.16 * L
            self.squash_ang += (-math.pi / 2 - self.squash_ang) * min(1, dt * 4)
        k, damp = lerp(70.0, 220.0, m), lerp(6.0, 10.0, m)
        self.squash_v += (-k * (self.squash - rest) - damp * self.squash_v) * dt
        self.squash += self.squash_v * dt
        self.squash = max(-0.3, min(0.45, self.squash))

        # googly pupils: spring with inertia; lazy eyes sag and drift slowly
        stiff = lerp(12.0, 45.0, m)
        for p in self.pupils:
            p[2] += (-p[0] * stiff - p[2] * 2.5 - self.vx * 0.02) * dt
            p[3] += (-p[1] * stiff - p[3] * 2.5 - self.vy * 0.02 + 8 + 14 * L) * dt
            p[0] += p[2] * dt * 10
            p[1] += p[3] * dt * 10
            d = math.hypot(p[0], p[1])
            if d > 1:
                p[0] /= d; p[1] /= d
                p[2] *= -0.5; p[3] *= -0.5

        # blink: slow heavy blinks when lazy, twitchy when crazy
        if self.blink > 0:
            self.blink = max(0, self.blink - dt * lerp(2.0, 12.0, m))
        elif random.random() < dt * (0.25 + 2.5 * C):
            self.blink = 1.0

        # yawns when the music is mellow
        if self.yawn > 0:
            self.yawn = max(0, self.yawn - dt / 2.8)
        elif L > 0.6 and not quiet and random.random() < dt * 0.07 * (1 - M):
            self.yawn = 1.0   # bored, not when it's enjoying a mellow song

        # mellow: now and then closes its eyes in bliss for a few seconds
        if self.bliss > 0:
            self.bliss = max(0, self.bliss - dt / 3.5)
        elif M > 0.6 and self.yawn == 0 and random.random() < dt * 0.08:
            self.bliss = 1.0

        self.third_eye = max(0, self.third_eye - dt * 0.5)
        if C > 0.7:
            self.third_eye = max(self.third_eye, C)
        self.hue += dt * (0.01 + 0.06 * m + high * 0.15 * (0.3 + m) + self.charge * 0.5)
        if C > 0.5 and random.random() < dt * C * 3:
            self.hue += 0.33   # strobe

        # trail: short when lazy, long rainbow smear when crazy
        self.trail.append((self.x, self.y, r, self.hue))
        while len(self.trail) > int(lerp(6, TRAIL_MAX, m)):
            self.trail.pop(0)

        # continuous sparkles on treble
        if high > 0.35 and random.random() < high * 0.6 * (0.2 + 1.2 * m):
            ang = random.uniform(0, TAU)
            self.emit(Particle(
                self.x + math.cos(ang) * r, self.y + math.sin(ang) * r,
                math.cos(ang) * 120, math.sin(ang) * 120 - 60,
                random.uniform(0.4, 0.9), self.hue + random.uniform(0.3, 0.7), random.uniform(2, 5), "spark"))

        # heavy: blood drips from the eyes and falls off the chin
        if self.evil > 0.3 and random.random() < dt * 2.5 * self.evil:
            side = random.choice((-0.36, 0.34))
            self.particles.append(Particle(
                self.x + side * r + random.uniform(-0.05, 0.05) * r, self.y + r * 0.85,
                self.vx * 0.3, random.uniform(0, 60), random.uniform(0.6, 1.0), 0.0,
                random.uniform(2.5, 4.5), "blood"))

        # mellow: little music notes drift up from the blob
        if M > 0.3 and level > 0.1 and random.random() < dt * 2.0 * M:
            self.emit(Particle(
                self.x + random.uniform(-0.6, 0.6) * r, self.y - r * 0.8,
                random.uniform(-25, 25), random.uniform(-70, -40),
                random.uniform(2.0, 3.0), self.hue + random.uniform(0.3, 0.7), r * 0.3, "note"))

        for p in self.particles:
            p.life -= dt
            if p.kind == "note":
                p.vx += math.sin(self.t * 2 + p.rot) * 40 * dt   # lazy wobble, no gravity
            elif p.kind == "smoke":
                p.vx += math.sin(self.t * 1.5 + p.rot) * 30 * dt   # curls as it rises
                p.size += dt * 9
            elif p.kind != "ring":
                p.vy += 500 * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.rot += dt * 6
        self.particles = [p for p in self.particles if p.life > 0]

    def dark(self, rgb, k=0.85):
        """Fade a colour toward DARK as the music gets heavy."""
        t = self.heavy * k
        return tuple(c + (d - c) * t for c, d in zip(rgb, DARK))

    def emit(self, p):
        """Spawn a confetti particle, thinned by the confetti budget."""
        if len(self.particles) < self.max_particles and random.random() < self.confetti_rate:
            self.particles.append(p)

    def radius(self, bass):
        # inflates while a build-up charges
        return self.R * (1 + min(bass, 2.0) * 0.18) * (1 + 0.3 * self.charge)

    def on_kick(self, strength, W, H, a):
        self.penta_pulse = max(self.penta_pulse, 0.6)
        if self.hover > 0.5:
            # floating in a breakdown: kicks just make it pulse
            self.squash_v += 1.5
            return
        self.jump(strength, W, H, a, synced=True)

    def on_snare(self, strength):
        """Headbang! Body nods down, face dips, little sideways shove."""
        s = min(strength, 3.0)
        self.penta_pulse = 1.0
        self.nod = min(1.0, 0.5 + 0.25 * s)
        self.squash_v += 1.5 + s * (0.5 + self.horns)
        self.squash_ang = -math.pi / 2
        self.nod_side = -self.nod_side
        self.vx += self.nod_side * 120 * (0.5 + self.mood)
        if self.hover > 0.3:            # snare rolls in a build-up: crackle
            for _ in range(3):
                ang = random.uniform(0, TAU)
                self.emit(Particle(
                    self.x + math.cos(ang) * self.R, self.y + math.sin(ang) * self.R,
                    math.cos(ang) * 200, math.sin(ang) * 200, 0.4, self.hue + 0.5, 3, "spark"))

    def on_hat(self):
        """Hi-hats (16ths in techno/psy): tiny sparks fizzing off the body."""
        if random.random() < 0.3 + 0.5 * self.mood:
            ang = random.uniform(0, TAU)
            r = self.R
            self.emit(Particle(
                self.x + math.cos(ang) * r, self.y + math.sin(ang) * r,
                math.cos(ang) * 160, math.sin(ang) * 160 - 40,
                random.uniform(0.25, 0.5), self.hue + random.uniform(0.3, 0.7), random.uniform(1.5, 3.5), "spark"))

    def on_drop(self, strength, W, H):
        """The kick is back after a breakdown: EXPLODE."""
        s = min(strength, 3.0)
        self.drop_boost = 1.0
        self.hover = 0.0
        self.land_at = 0.0
        self.charge = 0.0
        self.spiral = 0.0
        self.flash = min(1.0, 0.4 + 0.2 * s)
        self.third_eye = 1.0
        self.vy = -math.sqrt(2 * 2500 * H * 0.75)
        self.vx = random.choice((-1, 1)) * random.uniform(600, 1100)
        self.squash_v -= 8
        self.squash_ang = -math.pi / 2
        self.hue += 0.5
        for _ in range(int(60 + 40 * s)):
            ang = random.uniform(0, TAU)
            sp = random.uniform(200, 1100)
            kind = random.choice(("dot", "star", "star", "tri", "squig"))
            self.emit(Particle(
                self.x, self.y, math.cos(ang) * sp, math.sin(ang) * sp,
                random.uniform(0.8, 2.0), self.hue + random.random(), random.uniform(5, 14), kind))
        for i in range(4):
            self.particles.append(Particle(self.x, self.y, 0, 0, 0.5 + i * 0.25, self.hue + i * 0.25,
                                           self.R * (0.6 + i * 0.5), "ring"))

    def jump(self, strength, W, H, a, synced):
        s = min(strength, 2.5)
        m, L, C = self.mood, self.lazy, self.crazy
        # too lazy (or too mellow) to jump for weak beats: just a squishy little bop in place
        L = max(L, self.mellow)
        if L > 0 and s < 1.2 + 0.8 * L and random.random() < 0.3 + 0.6 * L:
            self.squash_v += 1.5 + s
            self.squash_ang = -math.pi / 2
            return
        r = self.R
        now = time.monotonic()
        if synced and self.land_at - now > 0.08 and not self.on_floor:
            # mid-flight on a planned beat-synced arc: don't break the rhythm
            self.squash_v -= 0.8 + 0.4 * s
            self.burst(int(3 + 3 * s * m), s * 0.5)
            return
        if self.vy < -250:
            # already flying up: kicks in mid-air just squish and sparkle
            self.squash_v -= 1.0 + 0.5 * s
            self.burst(int(4 + 4 * s * m), s * 0.6)
            return
        # jump height scales with the screen and the mood
        height = H * (0.18 + 0.22 * s / 2.5) * random.uniform(0.8, 1.15) * lerp(0.45, 1.4, m)
        P = a.beat_period if synced else 0.0
        floor_y = H - r
        vy = None
        if P:
            # steady tempo (techno/psy/steady rock): choose an airtime of whole
            # beats so the blob lands exactly on a future kick
            best = None
            for n in (1, 2, 3, 4):
                T = n * P
                v = (floor_y - self.y - 0.5 * self.g * T * T) / T
                apex = floor_y - (self.y - v * v / (2 * self.g) if v < 0 else self.y)
                if v >= 0 or apex > H - 2.4 * r:
                    continue
                cost = abs(math.log(max(apex, 1) / height))
                if best is None or cost < best[0]:
                    best = (cost, v, T)
            if best:
                vy = best[1]
                self.land_at = now + best[2]
        if vy is None:
            vy = -math.sqrt(2 * self.g * height)
        self.vy = vy
        # aim loosely away from walls so it roams the whole desktop
        bias = (W / 2 - self.x) / (W / 2)
        self.vx += random.uniform(-450, 450) * lerp(0.35, 1.8, m) + bias * 300
        self.squash_v -= (2 + s * 1.2) * lerp(0.6, 1.4, m)   # stretch upward
        self.squash_ang = -math.pi / 2
        self.hue += random.choice((0.08, 0.17, 0.33, 0.5)) * (1 if s > 1.6 else 0.4) * (0.3 + m)
        if s > 2.2 or (C > 0.5 and s > 1.5):
            self.third_eye = 1.0
        self.burst(int((10 + 16 * s) * lerp(0.25, 1.8, m)), s)
        self.particles.append(Particle(self.x, self.y, 0, 0, 0.5, self.hue + 0.5, self.R, "ring"))
        if C > 0.6:
            self.particles.append(Particle(self.x, self.y, 0, 0, 0.35, self.hue + 0.25, self.R * 0.6, "ring"))

    def burst(self, n, s):
        m = self.mood
        for _ in range(n):
            ang = random.uniform(0, TAU)
            sp = random.uniform(120, 520) * (0.6 + s * 0.3) * lerp(0.6, 1.3, m)
            kind = random.choice(("dot", "dot", "star", "tri", "squig"))
            self.emit(Particle(
                self.x, self.y, math.cos(ang) * sp, math.sin(ang) * sp,
                random.uniform(0.6, 1.5), self.hue + random.random(), random.uniform(4, 11), kind))

    def splash(self, speed, ang, r):
        n = int(min(speed / 120, 14))
        nx, ny = -math.cos(ang), -math.sin(ang)  # direction toward the wall
        for _ in range(n):
            spread = random.uniform(-1.2, 1.2)
            dx = -nx * math.cos(spread) - (-ny) * math.sin(spread)
            dy = -ny * math.cos(spread) + (-nx) * math.sin(spread)
            sp = random.uniform(100, 350)
            self.emit(Particle(
                self.x + nx * r, self.y + ny * r, dx * sp, dy * sp,
                random.uniform(0.3, 0.7), self.hue + 0.5, random.uniform(3, 6), "dot"))

    # ------------------------------------------------------------------ drawing
    def outline(self, a, r):
        pts = []
        spec = a.spectrum
        m, C = self.mood, self.crazy
        at = self.anim_t
        speed = math.hypot(self.vx, self.vy)
        stretch = min(speed / 7000, 0.15)
        vang = math.atan2(self.vy, self.vx)
        sq = self.squash
        wob_amp = 0.5 + m
        ca, sa = math.cos(self.squash_ang), math.sin(self.squash_ang)
        cv, sv2 = math.cos(vang), math.sin(vang)
        for i in range(self.N):
            th = i / self.N * TAU
            # spectrum mapped symmetrically around the body, rotating slowly
            u = abs(((th + at * 0.6) % TAU) / TAU * 2 - 1)
            sv = spec[min(int(u * 31.99), 31)]
            wob = (math.sin(th * 3 + at * 2.3) * 0.05 +
                   math.sin(th * 5 - at * 3.1) * 0.035 * (1 + a.mid) +
                   math.sin(th * 9 + at * 7.0) * 0.02 * a.high) * wob_amp
            # crazy: spiky, buzzing outline
            wob += math.sin(th * 13 + self.t * 30) * 0.07 * C * (0.5 + a.high)
            rr = r * (1 + sv * 0.32 * min(a.level * 2, 1) * (0.5 + m) + wob)
            px, py = math.cos(th) * rr, math.sin(th) * rr
            # squash along impact axis
            pa = px * ca + py * sa
            pp = -px * sa + py * ca
            pa *= (1 - sq)
            pp *= (1 + sq * 0.6)
            px, py = pa * ca - pp * sa, pa * sa + pp * ca
            # stretch along velocity
            pa = px * cv + py * sv2
            pp = -px * sv2 + py * cv
            pa *= (1 + stretch)
            pp *= (1 - stretch * 0.5)
            px, py = pa * cv - pp * sv2, pa * sv2 + pp * cv
            pts.append((self.x + px, self.y + py))
        return pts

    def draw(self, cr, a):
        r = self.radius(a.bass)
        sat = lerp(0.6, 1.0, clamp01(self.mood * 2))

        # psychedelic ghost trail
        n = len(self.trail)
        for i, (tx, ty, tr, th) in enumerate(self.trail[:-1]):
            f = i / n
            cr.set_source_rgba(*self.dark(hsv(th + f * 0.5 + 0.3, sat, 1)), f * 0.22)
            cr.arc(tx, ty, tr * (0.35 + f * 0.6), 0, TAU)
            cr.fill()

        # particles behind (smoke rises in front of the face, drawn last)
        for p in self.particles:
            if p.kind != "smoke":
                self.draw_particle(cr, p)

        cr.save()
        cr.translate(*self.shake)
        # mellow: rocks its head side to side
        tilt = 0.14 * self.mellow * math.sin(self.sway)
        if abs(tilt) > 1e-3:
            cr.translate(self.x, self.y)
            cr.rotate(tilt)
            cr.translate(-self.x, -self.y)
        pts = self.outline(a, r)

        # glow (bigger and brighter while a build-up charges)
        for k, al in ((1.45 + 0.4 * self.charge, 0.06 + 0.12 * self.charge), (1.22, 0.10)):
            cr.save()
            cr.translate(self.x, self.y)
            cr.scale(k, k)
            cr.translate(-self.x, -self.y)
            smooth_closed_path(cr, pts)
            glow = hsv(self.hue + 0.5, sat)
            glow = tuple(lerp(c, d, self.heavy) for c, d in zip(glow, (0.75, 0.0, 0.05)))
            cr.set_source_rgba(*glow, al + a.level * 0.08 + self.crazy * 0.08)
            cr.fill()
            cr.restore()

        # body
        smooth_closed_path(cr, pts)
        grad = cairo.RadialGradient(self.x - r * 0.35, self.y - r * 0.4, r * 0.05,
                                    self.x, self.y, r * 1.35)
        grad.add_color_stop_rgba(0, *self.dark(hsv(self.hue + 0.12, 0.35 * sat, 1), 0.75), 1)
        grad.add_color_stop_rgba(0.45, *self.dark(hsv(self.hue, sat, 1)), 1)
        grad.add_color_stop_rgba(1, *self.dark(hsv(self.hue - 0.18, sat, 0.55), 1.0), 1)
        cr.set_source(grad)
        cr.fill_preserve()

        # swirling lava spots inside (clipped)
        cr.save()
        cr.clip_preserve()
        at = self.anim_t
        for j in range(6):
            ang = self.spin * (1 if j % 2 else -0.7) + j * TAU / 6 + math.sin(at + j) * 0.5
            dist = r * (0.35 + 0.25 * math.sin(at * 1.3 + j * 2))
            sr = r * (0.18 + 0.12 * a.spectrum[(j * 5) % 32])
            cr.set_source_rgba(*self.dark(hsv(self.hue + 0.25 + j * 0.13, 0.9 * sat, 1), 0.9), 0.55)
            cr.arc(self.x + math.cos(ang) * dist, self.y + math.sin(ang) * dist, sr, 0, TAU)
            cr.fill()
        cr.restore()

        cr.set_line_width(4 + a.high * 4)
        cr.set_source_rgba(*hsv(self.hue + 0.5, 1, 0.25), 0.9)
        cr.stroke()

        # shine (dull when dark)
        cr.set_source_rgba(1, 1, 1, 0.35 * (1 - 0.6 * self.heavy))
        cr.save()
        cr.translate(self.x - r * 0.35, self.y - r * 0.45)
        cr.rotate(-0.6)
        cr.scale(1, 0.55)
        cr.arc(0, 0, r * 0.22, 0, TAU)
        cr.fill()
        cr.restore()

        if self.horns > 0.02:
            self.draw_horns(cr, r)
        self.draw_face(cr, a, r)
        cr.restore()
        for p in self.particles:
            if p.kind == "smoke":
                self.draw_particle(cr, p)

    def draw_flash(self, cr):
        """Full-screen colour flash on a drop."""
        if self.flash > 0.01:
            cr.set_source_rgba(*hsv(self.hue + 0.5), 0.22 * self.flash)
            cr.paint()

    def draw_horns(self, cr, r):
        """Devil horns that grow out of the head when the guitars get distorted."""
        h = self.horns
        nod = self.nod
        for side in (-1, 1):
            ang = -math.pi / 2 + side * (0.62 + 0.1 * nod)
            bx = self.x + math.cos(ang) * r * 0.7
            by = self.y + math.sin(ang) * r * 0.7 + nod * r * 0.1
            length = r * 0.95 * h
            # curve outward then back in at the tip
            tx = bx + side * length * 0.55
            ty = by - length
            w = r * 0.2 * (0.4 + 0.6 * h)
            cr.move_to(bx - w, by + w * 0.3)
            cr.curve_to(bx - w + side * length * 0.1, by - length * 0.5,
                        tx - side * length * 0.25, ty + length * 0.15, tx, ty)
            cr.curve_to(tx - side * length * 0.05, ty + length * 0.35,
                        bx + w + side * length * 0.25, by - length * 0.4, bx + w, by + w * 0.3)
            cr.close_path()
            grad = cairo.LinearGradient(bx, by, tx, ty)
            grad.add_color_stop_rgba(0, 0.55, 0.02, 0.05, 1)
            grad.add_color_stop_rgba(1, 1.0, 0.25 + 0.3 * self.crazy, 0.1, 1)
            cr.set_source(grad)
            cr.fill_preserve()
            cr.set_line_width(3)
            cr.set_source_rgba(0.15, 0, 0.05, 0.9)
            cr.stroke()

    def draw_face(self, cr, a, r):
        m, L, C = self.mood, self.lazy, self.crazy
        M = self.mellow
        sq = self.squash
        face_y = (self.y - r * 0.12 * (1 - sq) + r * 0.08 * L   # lazy face slides down
                  + self.nod * r * 0.14)                       # headbang dip
        sleepy = self.sleepiness
        yawn_open = math.sin(math.pi * (1 - self.yawn)) if self.yawn > 0 else 0.0

        eyes = ((-0.36, -0.08, self.eye_sizes[0]), (0.34, -0.02, self.eye_sizes[1]))
        # droopy half-closed lids when lazy, wide open when crazy; squint on the headbang
        # mellow: relaxed, content lids (not droopy), fully shut while in bliss
        bliss = math.sin(math.pi * (1 - self.bliss)) ** 0.3 if self.bliss > 0 else 0.0
        lid = max(self.blink, sleepy * 0.9, lerp(L * 0.5, 0.3, M), yawn_open * 0.85,
                  self.nod * 0.45, bliss) * (1 - self.spiral)

        # rosy cheeks
        if M > 0.02:
            for cx in (-0.58, 0.56):
                cr.save()
                cr.translate(self.x + cx * r, face_y + r * 0.27)
                cr.scale(1, 0.6)
                cr.arc(0, 0, r * 0.15, 0, TAU)
                cr.restore()
                cr.set_source_rgba(1.0, 0.3, 0.45, 0.6 * M)
                cr.fill()

        for i, ((ex, ey, es), p) in enumerate(zip(eyes, self.pupils)):
            grow = 1 + 0.3 * C * (1 + 0.3 * math.sin(self.t * 20 + i * 2))
            ex_ = self.x + ex * r
            ey_ = face_y + ey * r
            er = r * 0.27 * es * grow
            self.draw_eye(cr, ex_, ey_, er, p, lid, a, idx=i)
            if self.evil > 0.05:
                self.draw_blood_tear(cr, ex_, ey_, er, r, i)
            if bliss > 0.5:
                # happy closed eyes: ^ ^
                cr.arc(ex_, ey_ + er * 0.45, er * 0.6, math.pi + 0.5, TAU - 0.5)
                cr.set_line_cap(cairo.LINE_CAP_ROUND)
                cr.set_line_width(max(2.5, er * 0.16))
                cr.set_source_rgba(0.1, 0, 0.1, (bliss - 0.5) * 2)
                cr.stroke()
            self.draw_brow(cr, ex_, ey_, er, r, side=-1 if i == 0 else 1, idx=i)

        # forehead: the third eye gives way to a pentagram when it gets heavy
        third = self.third_eye * clamp01(1 - self.heavy * 2)
        if third > 0.01:
            s = min(1, third * 2)
            self.draw_eye(cr, self.x, face_y - r * 0.5, r * 0.16 * s, self.pupils[0], 0, a,
                          hue_eye=True, idx=2)
        if self.heavy > 0.15:
            self.draw_pentagram(cr, self.x, face_y - r * 0.7, r * 0.2)

        # mouth: opens with volume; small & lopsided when lazy, huge grin when crazy
        # mellow: a wide, gentle, mostly closed smile instead of the lazy "meh"
        mw = r * 0.42 * lerp(0.6, 1.25, m) * (1 - yawn_open * 0.45) * (1 + 0.3 * M)
        mh = r * (0.04 + min(a.level, 1.2) * 0.32) * lerp(0.35, 1.5, m) * (1 - sleepy) * (1 - 0.5 * M)
        mh = max(mh, r * 0.55 * yawn_open)
        mh *= 1 - 0.75 * self.cool                # too cool to open its mouth much
        my = face_y + r * 0.42
        tilt = r * 0.12 * L * (1 - M)            # one corner droops: "meh"
        smile = r * 0.13 * M                     # corners curl up

        def bend(u):
            return -smile * (2 * u - 1) ** 2
        steps = 16
        cr.save()
        cr.move_to(self.x - mw, my - tilt * 0.5 - smile)
        for i in range(steps + 1):
            u = i / steps
            x = self.x - mw + 2 * mw * u
            y = (my + (u - 0.5) * tilt + bend(u) + math.sin(u * math.pi) * mh +
                 math.sin(u * 18 + self.t * 20) * a.high * r * 0.03 * (0.3 + m))
            cr.line_to(x, y)
        for i in range(steps, -1, -1):
            u = i / steps
            x = self.x - mw + 2 * mw * u
            cr.line_to(x, my + (u - 0.5) * tilt + bend(u)
                       - math.sin(u * math.pi) * mh * (0.25 + yawn_open * 0.6))
        cr.close_path()
        cr.set_source_rgba(0.15, 0, 0.12, 0.95)
        cr.fill_preserve()
        cr.clip()
        # tongue (flaps around when crazy)
        flap = math.sin(self.t * 15) * C * mw * 0.3
        cr.set_source_rgba(*hsv(self.hue + 0.85, 0.7, 1), 1)
        cr.arc(self.x + mw * 0.2 + flap, my + mh * 1.0, mw * 0.4, 0, TAU)
        cr.fill()
        # manic teeth
        if C > 0.15 and mh > r * 0.08:
            cr.set_source_rgba(1, 1, 1, min(1, C * 2))
            nt = 7
            for i in range(nt):
                x0 = self.x - mw + 2 * mw * i / nt
                x1 = x0 + 2 * mw / nt
                top = my - mh * 0.3
                cr.move_to(x0, top)
                cr.line_to(x1, top)
                cr.line_to((x0 + x1) / 2, top + mh * 0.45)
                cr.close_path()
                cr.fill()
        cr.restore()

        if self.cool > 0.01:
            self.draw_cigarette(cr, self.x + mw * 0.75, my + mh * 0.3, r)
            self.draw_sunglasses(cr, face_y, r)
        else:
            self.ember = None

        if sleepy > 0.8:
            cr.select_font_face("Sans", cairo.FONT_SLANT_ITALIC, cairo.FONT_WEIGHT_BOLD)
            for i in range(3):
                ph = (self.t * 0.5 + i / 3) % 1
                cr.set_font_size(r * (0.2 + ph * 0.25))
                cr.set_source_rgba(*hsv(self.hue + i * 0.2), 1 - ph)
                cr.move_to(self.x + r * (0.6 + ph * 0.6), self.y - r * (0.7 + ph * 1.1))
                cr.show_text("z")

    def draw_sunglasses(self, cr, face_y, r):
        """Easter egg: dark shades that slide down onto the face."""
        k = self.cool
        y = face_y - r * 0.05 - (1 - k) ** 2 * r * 1.2      # drops in from above
        lw, lh = r * 0.34, r * 0.48                         # big enough to hide both googly eyes
        cr.save()
        cr.set_line_join(cairo.LINE_JOIN_ROUND)
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        # arms and bridge
        cr.set_source_rgba(0.06, 0.06, 0.07, k)
        cr.set_line_width(max(2.5, r * 0.06))
        cr.move_to(self.x - r * 0.37 - lw, y - lh * 0.5)
        cr.line_to(self.x - r * 0.92, y - lh * 0.7)
        cr.move_to(self.x + r * 0.37 + lw, y - lh * 0.5)
        cr.line_to(self.x + r * 0.92, y - lh * 0.7)
        cr.move_to(self.x - r * 0.06, y - lh * 0.45)
        cr.curve_to(self.x - r * 0.02, y - lh * 0.75, self.x + r * 0.02, y - lh * 0.75,
                    self.x + r * 0.06, y - lh * 0.45)
        cr.stroke()
        for side in (-1, 1):
            cx = self.x + side * r * 0.37
            # wayfarer-ish lens: flat top, rounded bottom
            cr.move_to(cx - lw, y - lh * 0.65)
            cr.line_to(cx + lw, y - lh * 0.65)
            cr.curve_to(cx + lw, y + lh * 0.5, cx + lw * 0.5, y + lh * 0.75, cx, y + lh * 0.75)
            cr.curve_to(cx - lw * 0.5, y + lh * 0.75, cx - lw, y + lh * 0.5, cx - lw, y - lh * 0.65)
            cr.close_path()
            g = cairo.LinearGradient(cx, y - lh, cx, y + lh)
            g.add_color_stop_rgba(0, 0.18, 0.18, 0.22, k)
            g.add_color_stop_rgba(1, 0.02, 0.02, 0.03, k)
            cr.set_source(g)
            cr.fill_preserve()
            cr.set_source_rgba(0.04, 0.04, 0.05, k)
            cr.set_line_width(max(2, r * 0.05))
            cr.stroke()
            # glint
            cr.move_to(cx - lw * 0.6, y - lh * 0.35)
            cr.line_to(cx - lw * 0.15, y - lh * 0.35)
            cr.line_to(cx - lw * 0.45, y + lh * 0.2)
            cr.close_path()
            cr.set_source_rgba(1, 1, 1, 0.3 * k)
            cr.fill()
        cr.restore()

    def draw_cigarette(self, cr, x, y, r):
        """Easter egg: a cigarette hanging from the corner of the mouth, ember glowing."""
        k = self.cool
        ang = 0.35 + 0.05 * math.sin(self.t * 0.9)         # droops a little, bobs lazily
        length, w = r * 0.5 * k, r * 0.075
        dx, dy = math.cos(ang), math.sin(ang)
        nx, ny = -dy, dx
        tx, ty = x + dx * length, y + dy * length

        def seg(t0, t1):
            ax, ay = x + dx * length * t0, y + dy * length * t0
            bx, by = x + dx * length * t1, y + dy * length * t1
            cr.move_to(ax + nx * w, ay + ny * w)
            cr.line_to(bx + nx * w, by + ny * w)
            cr.line_to(bx - nx * w, by - ny * w)
            cr.line_to(ax - nx * w, ay - ny * w)
            cr.close_path()

        cr.save()
        seg(0, 0.3)                                         # filter
        cr.set_source_rgba(0.85, 0.55, 0.25, k)
        cr.fill()
        seg(0.3, 0.9)                                       # paper
        cr.set_source_rgba(0.97, 0.96, 0.92, k)
        cr.fill()
        seg(0.9, 1.0)                                       # ash
        cr.set_source_rgba(0.45, 0.43, 0.42, k)
        cr.fill()
        # ember: breathes slowly, like someone taking drags
        drag = 0.5 + 0.5 * math.sin(self.t * 0.8) ** 8
        cr.arc(tx, ty, w * (1.6 + 0.8 * drag), 0, TAU)
        cr.set_source_rgba(1.0, 0.35, 0.05, 0.35 * drag * k)
        cr.fill()
        cr.arc(tx, ty, w * 0.75, 0, TAU)
        cr.set_source_rgba(1.0, 0.3 + 0.4 * drag, 0.05, k)
        cr.fill()
        cr.restore()
        self.ember = (tx + self.shake[0], ty + self.shake[1])

    def draw_blood_tear(self, cr, x, y, er, r, idx):
        """A streak of blood running down from under the eye, a drop swelling at its end."""
        ev = self.evil
        # each eye's streak slowly grows and shrinks, out of step with the other
        grow = 0.75 + 0.25 * math.sin(self.t * 0.8 + idx * 2.1)
        length = r * 0.55 * ev * grow
        x0 = x + er * (0.15 if idx == 0 else -0.2)
        y0 = y + er * 0.8
        w = er * 0.16
        sway = math.sin(self.t * 1.3 + idx) * w * 0.6
        cr.move_to(x0 - w, y0)
        cr.curve_to(x0 - w * 0.6, y0 + length * 0.4, x0 - w * 0.5 + sway, y0 + length * 0.7,
                    x0 - w * 0.45 + sway, y0 + length)
        cr.arc_negative(x0 + sway, y0 + length, w * 0.45, math.pi, 0)
        cr.curve_to(x0 + w * 0.5 + sway, y0 + length * 0.7, x0 + w * 0.6, y0 + length * 0.4,
                    x0 + w, y0)
        cr.close_path()
        cr.set_source_rgba(0.85, 0.02, 0.05, 0.95 * ev)   # bright enough to read on the dark body
        cr.fill()
        # swelling drop at the tip
        drop = w * (0.7 + 0.5 * grow + 0.3 * self.penta_pulse)
        cr.arc(x0 + sway, y0 + length + drop * 0.4, drop, 0, TAU)
        cr.fill()
        # wet highlight
        cr.set_source_rgba(1, 0.5, 0.5, 0.35 * ev)
        cr.arc(x0 + sway - drop * 0.35, y0 + length + drop * 0.1, drop * 0.25, 0, TAU)
        cr.fill()

    def draw_pentagram(self, cr, x, y, size):
        """Inverted pentagram in a circle, glowing red; flares on kicks and snares."""
        vis = clamp01((self.heavy - 0.15) / 0.45)
        size *= 0.6 + 0.4 * vis
        pulse = self.penta_pulse
        # points: start at the bottom (inverted), then every second vertex
        pts = [(x + math.cos(math.pi / 2 + k * 2 * TAU / 5) * size,
                y + math.sin(math.pi / 2 + k * 2 * TAU / 5) * size) for k in range(5)]

        def shape():
            cr.new_path()
            cr.arc(x, y, size * 1.12, 0, TAU)
            cr.new_sub_path()
            cr.move_to(*pts[0])
            for p in pts[1:]:
                cr.line_to(*p)
            cr.close_path()

        cr.save()
        cr.set_line_join(cairo.LINE_JOIN_MITER)
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        # burnt-in glow underneath, then the hot core line
        shape()
        cr.set_line_width(size * (0.3 + 0.2 * pulse))
        cr.set_source_rgba(0.9, 0.05, 0.0, vis * (0.22 + 0.3 * pulse))
        cr.stroke()
        shape()
        cr.set_line_width(max(1.5, size * 0.1))
        cr.set_source_rgba(1.0, 0.15 + 0.55 * pulse, 0.05 + 0.25 * pulse, vis)
        cr.stroke()
        cr.restore()

    def draw_brow(self, cr, x, y, er, r, side, idx):
        """Eyebrows: low and drooping when lazy, high and twitching when crazy."""
        L, C, m = self.lazy, self.crazy, self.mood
        lift = er * (lerp(-0.1, 0.45, m) + 0.25 * C * math.sin(self.t * 25 + idx * 1.7))
        by = y - er * 1.3 - lift
        w = er * 0.9
        outer_dy = er * (0.45 * L * (1 - self.mellow) - 0.5 * C * (1 if idx == 0 else -0.4)
                         - 0.6 * self.heavy)                  # heavy: angry V
        ox, ix = x + side * w, x - side * w
        oy, iy = by + outer_dy, by - outer_dy * 0.3
        cr.move_to(ox, oy)
        cr.curve_to(ox - side * w * 0.5, by - er * 0.2, ix + side * w * 0.5, by - er * 0.2, ix, iy)
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        cr.set_line_width(max(3, r * 0.07))
        brow = hsv(self.hue + 0.5, 1, 0.2)
        cr.set_source_rgba(*(lerp(c, d, self.heavy) for c, d in zip(brow, (0.7, 0.04, 0.02))), 0.95)
        cr.stroke()

    def draw_eye(self, cr, x, y, er, p, lid, a, hue_eye=False, idx=0):
        C, L = self.crazy, self.lazy
        ev = 0.0 if hue_eye else self.evil      # heavy: dark, bloodshot eyes
        cr.arc(x, y, er, 0, TAU)
        if ev > 0:
            # whites sink to a dark blood-red, deepest at the rim
            g = cairo.RadialGradient(x, y, er * 0.1, x, y, er)
            g.add_color_stop_rgb(0, *(lerp(1, c, ev) for c in (0.3, 0.02, 0.03)))
            g.add_color_stop_rgb(1, *(lerp(1, c, ev) for c in (0.1, 0.0, 0.01)))
            cr.set_source(g)
        else:
            cr.set_source_rgba(1, 1, 1, 1)
        cr.fill_preserve()
        cr.set_line_width(max(2, er * 0.12))
        cr.set_source_rgba(0.1, 0, 0.1, 1)
        cr.stroke()
        if ev > 0.05:
            # bloodshot veins creeping in from the rim, throbbing with the drums
            cr.save()
            cr.set_line_cap(cairo.LINE_CAP_ROUND)
            cr.set_line_width(max(1.0, er * 0.06))
            cr.set_source_rgba(0.95, 0.08 + 0.2 * self.penta_pulse, 0.05, ev * (0.75 + 0.25 * self.penta_pulse))
            for ang, ln, kink in self.veins[idx % 2]:
                ln *= ev
                c, s_ = math.cos(ang), math.sin(ang)
                n_, m_ = math.cos(ang + kink), math.sin(ang + kink)
                cr.move_to(x + c * er * 0.97, y + s_ * er * 0.97)
                mx, my = x + c * er * (0.97 - ln * 0.5), y + s_ * er * (0.97 - ln * 0.5)
                cr.line_to(mx, my)
                cr.line_to(mx - n_ * er * ln * 0.5, my - m_ * er * ln * 0.5)
                cr.move_to(mx, my)            # little fork
                cr.line_to(mx - math.cos(ang - kink) * er * ln * 0.3,
                           my - math.sin(ang - kink) * er * ln * 0.3)
            cr.stroke()
            cr.restore()
        # pupils dilate with bass; tiny manic pinpricks when crazy, big & dopey when lazy
        pr = er * (0.42 + min(a.bass, 1.5) * 0.15) * lerp(1.0, 0.6, C) * (1 + 0.12 * L)
        # crazy: pupils spin in circles, in opposite directions
        dirn = 1 if idx % 2 == 0 else -1
        ox = lerp(p[0], math.cos(self.t * 14 * dirn), C * 0.85)
        oy = lerp(p[1], math.sin(self.t * 14 * dirn), C * 0.85)
        px = x + ox * (er - pr) * 0.95
        py = y + oy * (er - pr) * 0.95
        sp = self.spiral
        if sp < 0.98:
            if ev > 0.05:
                # glowing hot red pupils in the dark
                cr.arc(px, py, pr * 1.6, 0, TAU)
                cr.set_source_rgba(1.0, 0.1, 0.0, 0.3 * ev * (1 + self.penta_pulse) * (1 - sp))
                cr.fill()
            if hue_eye:
                cr.set_source_rgba(*hsv(self.hue + 0.5), 1 - sp)
            elif ev > 0.05:
                cr.set_source_rgba(lerp(0.55 * self.horns, 1.0, ev), 0.05 + 0.3 * self.penta_pulse * ev,
                                   0.02, 1 - sp)
            elif self.horns > 0.3:
                # metal: glowing red pupils
                cr.set_source_rgba(0.55 * self.horns, 0.0, 0.08 * (1 - self.horns), 1 - sp)
            else:
                cr.set_source_rgba(0.05, 0.0, 0.08, 1 - sp)
            cr.arc(px, py, pr, 0, TAU)
            cr.fill()
            cr.set_source_rgba(1, 1, 1, 0.9 * (1 - sp) * (1 - 0.6 * ev))
            cr.arc(px - pr * 0.35, py - pr * 0.35, pr * 0.28, 0, TAU)
            cr.fill()
        if sp > 0.02:
            # psy build-up: hypnotic rotating spirals, counter-rotating per eye
            cr.save()
            cr.arc(x, y, er * 0.95, 0, TAU)
            cr.clip()
            cr.translate(x, y)
            cr.rotate(self.t * (9 if idx % 2 == 0 else -9) * (1 + 2 * self.charge))
            cr.set_line_width(max(1.5, er * 0.13))
            cr.set_source_rgba(*hsv(self.hue + 0.5 + idx * 0.2, 1, 0.5), sp)
            steps = 60
            for i in range(steps):
                u = i / steps
                ang = u * TAU * 3
                rr = er * 0.95 * u
                (cr.move_to if i == 0 else cr.line_to)(math.cos(ang) * rr, math.sin(ang) * rr)
            cr.stroke()
            cr.restore()
        if lid > 0.02:
            cr.save()
            cr.arc(x, y, er + 1, 0, TAU)
            cr.clip()
            lid_h = (2 * er + 4) * lid
            cr.rectangle(x - er - 2, y - er - 2, 2 * er + 4, lid_h)
            cr.set_source_rgba(*self.dark(hsv(self.hue, 1, 0.8)), 1)
            cr.fill()
            # lid edge
            cr.move_to(x - er - 2, y - er - 2 + lid_h)
            cr.line_to(x + er + 2, y - er - 2 + lid_h)
            cr.set_line_width(max(2, er * 0.1))
            cr.set_source_rgba(0.1, 0, 0.1, 0.9)
            cr.stroke()
            cr.restore()

    def draw_particle(self, cr, p):
        f = p.life / p.max
        if p.kind == "ring":
            rad = p.size * (1 + (1 - f) * 3)
            cr.set_line_width(6 * f)
            cr.set_source_rgba(*hsv(p.hue), f * 0.8)
            cr.arc(p.x, p.y, rad, 0, TAU)
            cr.stroke()
            return
        if p.kind == "blood":
            cr.set_source_rgba(0.8, 0.02, 0.05, min(1, f * 2))
            s = p.size
            # teardrop stretched by its fall speed
            stretch = 1 + min(p.vy / 600, 1.5)
            cr.save()
            cr.translate(p.x, p.y)
            cr.scale(1, stretch)
            cr.move_to(0, -s * 1.8)
            cr.curve_to(s * 0.4, -s * 0.8, s, -s * 0.4, s, 0)
            cr.arc(0, 0, s, 0, math.pi)
            cr.curve_to(-s, -s * 0.4, -s * 0.4, -s * 0.8, 0, -s * 1.8)
            cr.restore()
            cr.fill()
            return
        if p.kind == "smoke":
            cr.set_source_rgba(0.82, 0.82, 0.85, 0.35 * f * min(1, (1 - f) * 6))
            cr.arc(p.x, p.y, p.size, 0, TAU)
            cr.fill()
            return
        if p.kind == "note":
            # fade in, then out
            # (drawn as shapes: cairo's toy text API has no font fallback for the glyph)
            cr.set_source_rgba(*hsv(p.hue, 0.6, 1), min(1, f * 1.5, (1 - f) * 5))
            s = p.size
            cr.save()
            cr.translate(p.x, p.y)
            cr.rotate(0.25 * math.sin(p.rot))
            double = p.hue % 0.2 < 0.1          # ♫ or ♪
            heads = (0.0, s * 0.9) if double else (0.0,)
            for hx in heads:
                cr.save()
                cr.translate(hx, 0)
                cr.rotate(-0.4)
                cr.scale(1, 0.72)
                cr.arc(0, 0, s * 0.3, 0, TAU)
                cr.restore()
                cr.fill()
                cr.rectangle(hx + s * 0.2, -s * 1.05, s * 0.09, s * 1.05)
                cr.fill()
            if double:
                cr.move_to(s * 0.2, -s * 1.05)
                cr.line_to(s * 1.19, -s * 1.2)
                cr.line_to(s * 1.19, -s * 0.98)
                cr.line_to(s * 0.2, -s * 0.83)
                cr.close_path()
            else:
                cr.move_to(s * 0.29, -s * 1.05)
                cr.curve_to(s * 0.35, -s * 0.7, s * 0.75, -s * 0.7, s * 0.6, -s * 0.3)
                cr.curve_to(s * 0.6, -s * 0.6, s * 0.4, -s * 0.7, s * 0.29, -s * 0.75)
                cr.close_path()
            cr.fill()
            cr.restore()
            return
        cr.set_source_rgba(*hsv(p.hue), min(1, f * 1.5))
        s = p.size * (0.5 + f * 0.5)
        if p.kind in ("dot", "spark"):
            cr.arc(p.x, p.y, s, 0, TAU)
            cr.fill()
        elif p.kind == "star":
            cr.save()
            cr.translate(p.x, p.y)
            cr.rotate(p.rot)
            for i in range(10):
                rr = s * (1.6 if i % 2 == 0 else 0.6)
                ang = i * TAU / 10
                (cr.move_to if i == 0 else cr.line_to)(math.cos(ang) * rr, math.sin(ang) * rr)
            cr.close_path()
            cr.fill()
            cr.restore()
        elif p.kind == "tri":
            cr.save()
            cr.translate(p.x, p.y)
            cr.rotate(p.rot)
            cr.move_to(0, -s * 1.4)
            cr.line_to(s * 1.2, s)
            cr.line_to(-s * 1.2, s)
            cr.close_path()
            cr.fill()
            cr.restore()
        else:  # squiggle
            cr.save()
            cr.translate(p.x, p.y)
            cr.rotate(p.rot)
            cr.set_line_width(max(1.5, s * 0.4))
            cr.move_to(-s * 2, 0)
            for i in range(9):
                u = i / 8
                cr.line_to(-s * 2 + u * s * 4, math.sin(u * TAU * 1.5) * s * 0.8)
            cr.stroke()
            cr.restore()


# --------------------------------------------------------------------------- window
class Overlay(Gtk.Window):
    def __init__(self, args, audio):
        super().__init__(title="soundblob")
        self.audio = audio
        self.blob = Blob(args.size, confetti=args.confetti)
        self.now_playing = NowPlaying()
        self.last_time = None
        # FPS counter: frames actually drawn, averaged over half-second windows
        self.show_fps = not args.no_fps
        self.show_scope = not args.no_scope
        self.show_vignette = not args.no_vignette
        self._vignette = None      # (W, H, surface): the gradient is rendered once per size
        self.scope_gain = 0.0
        self.fps = 0.0
        self.draw_ms = 0.0
        self._fps_frames = 0
        self._fps_draw = 0.0
        self._fps_since = time.perf_counter()

        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        if visual is not None:
            self.set_visual(visual)
        self.set_app_paintable(True)

        if GtkLayerShell is not None and GtkLayerShell.is_supported():
            GtkLayerShell.init_for_window(self)
            GtkLayerShell.set_namespace(self, "soundblob")
            layer = {"overlay": GtkLayerShell.Layer.OVERLAY, "top": GtkLayerShell.Layer.TOP,
                     "bottom": GtkLayerShell.Layer.BOTTOM}[args.layer]
            GtkLayerShell.set_layer(self, layer)
            for edge in (GtkLayerShell.Edge.TOP, GtkLayerShell.Edge.BOTTOM,
                         GtkLayerShell.Edge.LEFT, GtkLayerShell.Edge.RIGHT):
                GtkLayerShell.set_anchor(self, edge, True)
            GtkLayerShell.set_exclusive_zone(self, -1)
            GtkLayerShell.set_keyboard_mode(self, GtkLayerShell.KeyboardMode.NONE)
        else:
            print("layer-shell not available; falling back to a fullscreen window", file=sys.stderr)
            self.set_decorated(False)
            self.fullscreen()

        self.connect("draw", self.on_draw)
        self.connect("realize", self.on_realize)
        self.connect("destroy", Gtk.main_quit)
        self.add_tick_callback(self.on_tick)

    def on_realize(self, *_):
        # empty input region => completely click-through
        self.input_shape_combine_region(cairo.Region())

    def on_tick(self, widget, clock):
        now = clock.get_frame_time() / 1e6
        dt = 1 / 60 if self.last_time is None else min(now - self.last_time, 1 / 20)
        self.last_time = now
        W, H = self.get_allocated_width(), self.get_allocated_height()
        if W > 1 and H > 1:
            # 2 substeps for stable bounces
            for _ in range(2):
                self.blob.update(dt / 2, W, H, self.audio)
        self.blob.cool_target = 1.0 if self.now_playing.lanegan else 0.0
        self.queue_draw()
        return GLib.SOURCE_CONTINUE

    def on_draw(self, widget, cr):
        t0 = time.perf_counter()
        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.set_source_rgba(0, 0, 0, 0)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)
        if self.show_vignette:
            self.draw_vignette(cr)
        self.blob.draw_flash(cr)
        if self.show_scope:
            self.draw_scope(cr)
        self.blob.draw(cr, self.audio)
        if self.show_fps:
            self.draw_fps(cr)
        t1 = time.perf_counter()
        self._fps_frames += 1
        self._fps_draw += t1 - t0
        if t1 - self._fps_since >= 0.5:
            self.fps = self._fps_frames / (t1 - self._fps_since)
            self.draw_ms = self._fps_draw / self._fps_frames * 1000
            self._fps_frames, self._fps_draw, self._fps_since = 0, 0.0, t1
        return False

    def draw_vignette(self, cr):
        """Dark edges creeping in from the screen corners when the music gets heavy."""
        h = self.blob.heavy
        if h < 0.01:
            return
        W, H = self.get_allocated_width(), self.get_allocated_height()
        if self._vignette is None or self._vignette[:2] != (W, H):
            # a full-screen gradient is expensive in cairo, so render it once and reuse it
            surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
            c = cairo.Context(surf)
            c.translate(W / 2, H / 2)
            c.scale(W / 2, H / 2)                 # unit circle -> ellipse that fits the screen
            grad = cairo.RadialGradient(0, 0, 0, 0, 0, math.sqrt(2))
            grad.add_color_stop_rgba(0.0, *DARK, 0.0)
            grad.add_color_stop_rgba(0.45, *DARK, 0.0)
            grad.add_color_stop_rgba(0.75, *DARK, 0.45)
            grad.add_color_stop_rgba(1.0, *DARK, 1.0)
            c.set_source(grad)
            c.paint()
            self._vignette = (W, H, surf)
        cr.set_source_surface(self._vignette[2], 0, 0)
        cr.paint_with_alpha(VIGNETTE_ALPHA * h * (0.9 + 0.1 * self.blob.penta_pulse))

    def draw_scope(self, cr):
        """Oscilloscope: the latest ~23 ms of audio as a rainbow line across the middle."""
        W, H = self.get_allocated_width(), self.get_allocated_height()
        buf = self.audio.buf                      # grab the reference once (audio thread swaps it)
        n = SCOPE_SAMPLES
        # trigger on the latest rising zero crossing that still leaves n samples,
        # so the waveform stands still instead of sliding around
        head = buf[:len(buf) - n + 1]
        cross = np.flatnonzero((head[:-1] < 0) & (head[1:] >= 0))
        start = int(cross[-1]) + 1 if len(cross) else len(buf) - n
        wave = buf[start:start + n]
        # auto gain: loud parts swing about +-15% of the screen height
        target = H * 0.15 / max(self.audio.peak * 2.5, 0.01)
        self.scope_gain += (target - self.scope_gain) * 0.1
        pts = max(64, min(W // 6, n))             # ~1 point per 6 px: smooth enough, cheap
        idx = np.linspace(0, n - 1, pts).astype(int)
        ys = H / 2 - np.clip(wave[idx] * self.scope_gain, -H * 0.3, H * 0.3)
        xs = np.linspace(0, W, pts)

        # Rainbow made of solid-colour pieces: a gradient source or a wide glow stroke
        # costs 2-5x more CPU in cairo (measured), so the line stays crisp and thin.
        hue = self.blob.hue
        level = min(self.audio.level, 1.0)
        cr.save()
        cr.set_line_join(cairo.LINE_JOIN_BEVEL)
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        cr.set_line_width(2.0 + 1.5 * level)
        pieces = SCOPE_PIECES
        step = (pts - 1) / pieces
        for c in range(pieces):
            i0, i1 = int(c * step), int((c + 1) * step)
            cr.move_to(xs[i0], ys[i0])
            for k in range(i0 + 1, i1 + 1):
                cr.line_to(xs[k], ys[k])
            cr.set_source_rgba(*hsv(hue + c / pieces * 0.6), 0.95)
            cr.stroke()
        cr.restore()

    def draw_fps(self, cr):
        """Small readout in the bottom-right corner: fps, CPU draw time, confetti count."""
        W, H = self.get_allocated_width(), self.get_allocated_height()
        text = f"{self.fps:3.0f} fps  {self.draw_ms:4.1f} ms  {len(self.blob.particles):3d} confetti"
        cr.select_font_face("monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        cr.set_font_size(13)
        ext = cr.text_extents(text)
        pad = 7
        w, h = ext.x_advance + 2 * pad, 13 + 2 * pad
        x, y = W - w - 16, H - h - 16
        r = 6
        cr.new_path()
        cr.arc(x + w - r, y + r, r, -math.pi / 2, 0)
        cr.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
        cr.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
        cr.arc(x + r, y + r, r, math.pi, 1.5 * math.pi)
        cr.close_path()
        cr.set_source_rgba(0, 0, 0, 0.6)
        cr.fill()
        # green when smooth, amber when struggling, red when choppy
        if self.fps >= 50:
            cr.set_source_rgb(0.45, 1.0, 0.55)
        elif self.fps >= 30:
            cr.set_source_rgb(1.0, 0.8, 0.3)
        else:
            cr.set_source_rgb(1.0, 0.4, 0.4)
        cr.move_to(x + pad, y + pad + 11)
        cr.show_text(text)


def main():
    ap = argparse.ArgumentParser(description="A weird blob that bounces to your music.")
    ap.add_argument("--device", default="@DEFAULT_MONITOR@",
                    help="PulseAudio source to listen to (default: monitor of default sink). "
                         "List with: pactl list short sources")
    ap.add_argument("--size", type=float, default=70, help="blob base radius in px (default 70)")
    ap.add_argument("--layer", choices=("overlay", "top", "bottom"), default="overlay",
                    help="overlay = above everything; bottom = only on the bare desktop/wallpaper")
    ap.add_argument("--confetti", type=float, default=1.0,
                    help="confetti amount: 0 = none, 1 = default, 2 = double (more costs CPU)")
    ap.add_argument("--no-fps", action="store_true", help="hide the FPS readout")
    ap.add_argument("--no-scope", action="store_true", help="hide the oscilloscope line")
    ap.add_argument("--no-vignette", action="store_true",
                    help="don't darken the screen edges when the music gets heavy")
    ap.add_argument("--debug", action="store_true",
                    help="print what the analyzer hears (tempo, drums, mood...) once a second")
    args = ap.parse_args()

    audio = Audio(args.device)
    audio.start()
    if args.debug:
        def report():
            print(audio.debug_line(), flush=True)
            return GLib.SOURCE_CONTINUE
        GLib.timeout_add(1000, report)

    win = Overlay(args, audio)
    win.show_all()

    try:
        gi.require_version("GLibUnix", "2.0")
        from gi.repository import GLibUnix
        add_signal = GLibUnix.signal_add
    except (ValueError, ImportError):
        add_signal = GLib.unix_signal_add
    for sig in (signal.SIGINT, signal.SIGTERM):
        add_signal(GLib.PRIORITY_DEFAULT, sig, Gtk.main_quit)

    # The overlay never takes keyboard focus, so a hotkey sends a signal instead:
    #   pkill -USR1 -f '^python3 \S*soundblob\.py( |$)'   -> toggle the FPS readout
    def toggle_fps():
        win.show_fps = not win.show_fps
        return GLib.SOURCE_CONTINUE
    add_signal(GLib.PRIORITY_DEFAULT, signal.SIGUSR1, toggle_fps)

    #   pkill -USR2 -f '^python3 \S*soundblob\.py( |$)'   -> toggle the oscilloscope
    def toggle_scope():
        win.show_scope = not win.show_scope
        return GLib.SOURCE_CONTINUE
    add_signal(GLib.PRIORITY_DEFAULT, signal.SIGUSR2, toggle_scope)
    Gtk.main()
    audio.running = False


if __name__ == "__main__":
    main()
