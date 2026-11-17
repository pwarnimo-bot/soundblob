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
import json
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

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
MEMPHIS_HUES = (0.48, 0.78, 0.92, 0.14)   # 90s confetti: teal, purple, hot pink, yellow
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


def _plays_on(artists, title, album, bands, albums=(), songs=(), skip=()):
    """Does any of the artists match a whole band, an (artist, album) or an (artist, title),
    unless that (artist, album/title) is in skip?"""
    t, al = _norm(title), _norm(album)
    for n in {_norm(a) for a in artists}:
        if (n, al) in skip or (n, t) in skip:
            continue
        if n in bands or (n, al) in albums or (n, t) in songs:
            return True
    return False


def lanegan_on(artists, title, album=""):
    """Is Mark Lanegan singing on this track?"""
    if any("lanegan" in a.lower() for a in artists) or "lanegan" in title.lower():
        return True
    return _plays_on(artists, title, album, LANEGAN_BANDS, LANEGAN_ALBUMS, LANEGAN_SONGS)


# Layne Staley easter egg: Alice in Chains up to 1996 (not the William DuVall albums),
# Mad Season (minus the songs only Lanegan sings on), and his side projects.
LAYNE_BANDS = {"alice in chains", "mad season", "class of 99", "alice mudgarden"}
LAYNE_SKIP = {
    ("alice in chains", "black gives way to blue"),
    ("alice in chains", "the devil put dinosaurs here"),
    ("alice in chains", "rainier fog"),
    ("mad season", "locomotive"),
    ("mad season", "black book of fear"),
    ("mad season", "slip away"),
}


def layne_on(artists, title, album=""):
    """Is Layne Staley singing on this track?"""
    if any("staley" in a.lower() for a in artists):
        return True
    return _plays_on(artists, title, album, LAYNE_BANDS, skip=LAYNE_SKIP)


# Richard Wright easter egg: all of Pink Floyd except The Final Cut (he'd left), his solo
# work, and records he played on with David Gilmour and Syd Barrett.
WRIGHT_BANDS = {"pink floyd", "richard wright", "rick wright"}
WRIGHT_ALBUMS = {
    ("david gilmour", "on an island"),
    ("david gilmour", "live in gdansk"),
    ("david gilmour", "remember that night"),
    ("syd barrett", "barrett"),
}
WRIGHT_SONGS = {("david gilmour", "a boat lies waiting")}
WRIGHT_SKIP = {("pink floyd", "the final cut"), ("pink floyd", "hey hey rise up")}


def wright_on(artists, title, album=""):
    """Is Richard Wright playing on this track?"""
    return _plays_on(artists, title, album, WRIGHT_BANDS, WRIGHT_ALBUMS, WRIGHT_SONGS, WRIGHT_SKIP)


# Kurt Cobain easter egg: Nirvana (not the 60s British band of the same name), his solo
# demos, and a few records he guested on.
KURT_BANDS = {"nirvana", "kurt cobain"}
KURT_SONGS = {
    ("mark lanegan", "where did you sleep last night"),
    ("william s burroughs", "the priest they called him"),
}
KURT_SKIP = {("nirvana", al) for al in (
    "the story of simon simopath", "all of us", "local anaesthetic", "dedicated to markos iii",
    "songs of love and praise", "black flower", "orange and blue", "rainbow chaser")}


def kurt_on(artists, title, album=""):
    """Is Kurt Cobain on this track?"""
    return _plays_on(artists, title, album, KURT_BANDS, songs=KURT_SONGS, skip=KURT_SKIP)


# John Garcia easter egg: Kyuss and every band he has fronted since.
GARCIA_BANDS = {"kyuss", "sons of kyuss", "kyuss lives", "unida", "slo burn", "hermano",
                "vista chino", "john garcia", "john garcia and the band of gold"}


def tool_on(artists, title, album=""):
    """Is this Tool?"""
    return _plays_on(artists, title, album, {"tool"})


def korn_on(artists, title, album=""):
    """Is this Korn (or Jonathan Davis solo)?"""
    return _plays_on(artists, title, album, {"korn", "jonathan davis", "jdevil"})


def nin_on(artists, title, album=""):
    """Is Trent Reznor on this track (Nine Inch Nails and friends)?"""
    return _plays_on(artists, title, album, {
        "nine inch nails", "trent reznor", "trent reznor and atticus ross",
        "how to destroy angels"})


def linkin_on(artists, title, album=""):
    """Is this Linkin Park (or Chester's and Mike's other bands)?"""
    return _plays_on(artists, title, album, {
        "linkin park", "chester bennington", "dead by sunrise", "grey daze", "mike shinoda",
        "fort minor"})


def rage_on(artists, title, album=""):
    """Is this Rage Against the Machine (or Zack's and the band's other projects)?"""
    return _plays_on(artists, title, album, {
        "rage against the machine", "zack de la rocha", "one day as a lion", "prophets of rage"})


def acid_bath_on(artists, title, album=""):
    """Is this Acid Bath (or Dax Riggs's later bands, or Shrüm before it)?"""
    return _plays_on(artists, title, album, {
        "acid bath", "dax riggs", "agents of oblivion", "deadboy the elephantmen",
        "deadboy and the elephantmen", "shrüm", "shrum"})


def nola_on(artists, title, album=""):
    """Is this Down or Corrosion of Conformity (Pepper Keenan's NOLA / Southern sludge)?"""
    return _plays_on(artists, title, album, {"down", "corrosion of conformity"})


def garcia_on(artists, title, album=""):
    """Is John Garcia singing on this track?"""
    if any("john garcia" in a.lower() for a in artists):
        return True
    return _plays_on(artists, title, album, GARCIA_BANDS)


class ReleaseDates:
    """Finds out when a song first came out, so the blob knows which era it's dancing to.
    Spotify's MPRIS metadata has no date, so it asks MusicBrainz (the open music database, no
    account needed) in a background thread, once per song: the earliest release of any
    recording of that title by that artist or of the album it's on, whichever is earlier.
    Answers, including "not found", are cached in ~/.cache/soundblob/release-dates.json."""

    API = "https://musicbrainz.org/ws/2/"
    HEADERS = {"User-Agent": "soundblob/1.0", "Accept": "application/json"}

    def __init__(self, enabled=True):
        self.enabled = enabled
        base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
        self.path = os.path.join(base, "soundblob", "release-dates.json")
        try:
            with open(self.path) as f:
                self.cache = json.load(f)
        except (OSError, ValueError):
            self.cache = {}
        self.queue = collections.deque()
        self.queued = set()
        self.failed = {}                 # key -> when a lookup failed (network); retried later
        self.wake = threading.Event()
        if enabled:
            threading.Thread(target=self._worker, daemon=True).start()

    @staticmethod
    def key(artist, title):
        return f"{_norm(artist)}|{_norm(title)}"

    def get(self, artist, title, album):
        """'1991-09-24', '1991' or similar if known, '' if MusicBrainz doesn't know,
        None while it's being looked up."""
        k = self.key(artist, title)
        if k in self.cache:
            return self.cache[k]
        if (self.enabled and k not in self.queued
                and time.monotonic() - self.failed.get(k, -1e9) > 120):
            self.queued.add(k)
            self.queue.append((k, artist, title, album))
            self.wake.set()
        return None

    def _worker(self):
        while True:
            self.wake.wait()
            self.wake.clear()
            while self.queue:
                k, artist, title, album = self.queue.popleft()
                try:
                    self.cache[k] = self._lookup(artist, title, album)
                    self._save()
                except Exception:          # offline, timeout, rate limited, bad answer...
                    self.failed[k] = time.monotonic()
                self.queued.discard(k)
                time.sleep(1.1)            # MusicBrainz allows one request per second

    def _search(self, entity, query):
        url = f"{self.API}{entity}/?fmt=json&limit=100&query={urllib.parse.quote(query)}"
        req = urllib.request.Request(url, headers=self.HEADERS)
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                # 503 = "slow down": MusicBrainz is busy or we were a bit too fast
                if e.code not in (429, 503) or attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))

    @staticmethod
    def _q(text):
        return '"' + text.replace("\\", " ").replace('"', " ") + '"'

    def _lookup(self, artist, title, album):
        def plain(name):                   # 'Lithium - Remastered' -> 'Lithium'
            return name.split(" - ")[0].split(" (")[0].split(" [")[0].strip()

        def earliest(items):
            dates = [i.get("first-release-date") for i in items if i.get("score", 0) >= 95]
            return min((d for d in dates if d), default="")

        # popular songs have so many live versions that the original recording can be missing
        # from the results, so the album's first release is checked too; the earliest wins
        found = [earliest(self._search(
            "recording", f"recording:{self._q(plain(title))} AND artist:{self._q(artist)}")
            .get("recordings", []))]
        if album:
            time.sleep(1.1)
            found.append(earliest(self._search(
                "release-group", f"releasegroup:{self._q(plain(album))} AND artist:{self._q(artist)}")
                .get("release-groups", [])))
        return min((d for d in found if d), default="")

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.cache, f)
            os.replace(tmp, self.path)
        except OSError:
            pass


class NowPlaying:
    """Asks Spotify (over MPRIS on the session bus) what's playing, every couple of seconds.
    Easter eggs: Mark Lanegan gets the blob into sunglasses and a cigarette, Layne Staley into
    chains and red round glasses, Richard Wright turns it into a prism, and Kurt Cobain gets
    white shades that come off for X eyes when the chorus kicks in, and John Garcia sends it
    tripping through the desert. Tool opens its third eye inside a spinning fractal mandala, Korn
    gives it dreadlocks and a spiked collar on a leash, Nine Inch Nails glitches it, and Linkin
    Park puts on headphones, sprays graffiti paint and screams the choruses. Rage Against the
    Machine masks it with a bandana under a red star and kill-switch stutters it. Acid Bath
    drags it through the bayou: swamp-green, melting and dripping acid, draped in Spanish moss,
    bubbling inside, with black butterflies fluttering around that scatter when it gets loud.
    Down and Corrosion of Conformity grow it a big Southern beard, put a gold fleur-de-lis on
    its forehead, turn it whiskey-brown and set it swaggering through a haze of smoke."""

    BUS_NAME = "org.mpris.MediaPlayer2.spotify"

    def __init__(self, lookup_dates=True):
        self.artists = []
        self.title = ""
        self.album = ""
        self.playing = False
        self.dates = ReleaseDates(lookup_dates)
        self.track_start = time.monotonic()   # for the VHS counter
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
        artists = list(md.get("xesam:artist") or [])
        title = md.get("xesam:title") or ""
        if (artists, title) != (self.artists, self.title):
            self.track_start = time.monotonic()
        self.artists, self.title = artists, title
        self.album = md.get("xesam:album") or ""

    def _on_status(self, conn, res):
        self.playing = self._get(conn, res) == "Playing"

    @property
    def released(self):
        """When the playing song first came out ('1994-03-08', '1994', ...), or None."""
        if not (self.playing and self.artists and self.title):
            return None
        return self.dates.get(self.artists[0], self.title, self.album) or None

    @property
    def lanegan(self):
        return self.playing and lanegan_on(self.artists, self.title, self.album)

    @property
    def layne(self):
        return self.playing and layne_on(self.artists, self.title, self.album)

    @property
    def wright(self):
        return self.playing and wright_on(self.artists, self.title, self.album)

    @property
    def kurt(self):
        return self.playing and kurt_on(self.artists, self.title, self.album)

    @property
    def tool(self):
        return self.playing and tool_on(self.artists, self.title, self.album)

    @property
    def korn(self):
        return self.playing and korn_on(self.artists, self.title, self.album)

    @property
    def nin(self):
        return self.playing and nin_on(self.artists, self.title, self.album)

    @property
    def linkin(self):
        return self.playing and linkin_on(self.artists, self.title, self.album)

    @property
    def rage(self):
        return self.playing and rage_on(self.artists, self.title, self.album)

    @property
    def acid_bath(self):
        return self.playing and acid_bath_on(self.artists, self.title, self.album)

    @property
    def nola(self):
        return self.playing and nola_on(self.artists, self.title, self.album)

    @property
    def garcia(self):
        return self.playing and garcia_on(self.artists, self.title, self.album)


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
        self.layne = 0.0           # easter egg: chains + red round glasses (Layne Staley)
        self.layne_target = 0.0
        self.wright = 0.0          # easter egg: light through the blob like a prism (Richard Wright)
        self.wright_target = 0.0
        self.kurt = 0.0            # easter egg: white shades, X eyes in the loud parts (Kurt Cobain)
        self.kurt_target = 0.0
        self.grunge = 0.0          # loud-chorus level while Kurt is on (shades up, X eyes)
        self.desert = 0.0          # easter egg: psychedelic desert trip (John Garcia / Kyuss)
        self.desert_target = 0.0
        self.tool = 0.0            # easter egg: third eye, fractal mandala, golden spiral (Tool)
        self.tool_target = 0.0
        self.mandala = None        # offscreen images of the mandala (see draw_mandala)
        self.korn = 0.0            # easter egg: dreadlocks, spiked collar on a leash, big bounces
        self.korn_target = 0.0
        self.dread_sw = 0.0        # dreadlock swing (spring), sideways
        self.dread_v = 0.0
        self.leash_x = 300.0       # where the leash disappears off the top of the screen
        self.nin = 0.0             # easter egg: steel-grey, scanlines, colour fringes, glitches (NIN)
        self.nin_target = 0.0
        self.glitch = 0.0          # 1 -> 0 during a glitch burst (on kicks and snares)
        self.linkin = 0.0          # easter egg: headphones, graffiti splats, screams (Linkin Park)
        self.linkin_target = 0.0
        self.scream = 0.0          # loud chorus while Linkin Park is on
        self.rage = 0.0            # easter egg: bandana, red star, red & black, kill switch (RATM)
        self.rage_target = 0.0
        self.kill = 0.0            # seconds of kill-switch stutter left
        self.acid = 0.0            # easter egg: swamp green, melting, moss, butterflies (Acid Bath)
        self.acid_target = 0.0
        self.nola = 0.0            # easter egg: beard, fleur-de-lis, whiskey, swagger, haze (Down/COC)
        self.nola_target = 0.0
        self.beard_sw = 0.0        # beard swing (spring), sideways
        self.beard_v = 0.0
        # era awareness (release year from MusicBrainz): the 90s and the early 2000s
        self.era90 = 0.0           # backwards cap, Memphis-pattern confetti, VHS camcorder overlay
        self.era90_target = 0.0
        self.y2k = 0.0             # frosted tips, glossy sheen, sparkles, orange camera date stamp
        self.y2k_target = 0.0

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
        self.layne += (self.layne_target - self.layne) * min(1, dt * 1.5)
        self.wright += (self.wright_target - self.wright) * min(1, dt * 1.5)
        self.kurt += (self.kurt_target - self.kurt) * min(1, dt * 1.5)
        self.desert += (self.desert_target - self.desert) * min(1, dt * 0.8)
        self.tool += (self.tool_target - self.tool) * min(1, dt * 0.8)
        self.korn += (self.korn_target - self.korn) * min(1, dt * 1.5)
        self.nin += (self.nin_target - self.nin) * min(1, dt * 1.5)
        self.glitch = max(0.0, self.glitch - dt * 5)
        self.linkin += (self.linkin_target - self.linkin) * min(1, dt * 1.5)
        self.rage += (self.rage_target - self.rage) * min(1, dt * 1.5)
        self.kill = max(0.0, self.kill - dt)
        self.acid += (self.acid_target - self.acid) * min(1, dt * 1.2)
        self.nola += (self.nola_target - self.nola) * min(1, dt * 1.2)
        self.era90 += (self.era90_target - self.era90) * min(1, dt * 1.5)
        self.y2k += (self.y2k_target - self.y2k) * min(1, dt * 1.5)
        self.scream = self.linkin * clamp01((self.grunge - 0.45) / 0.4)
        if self.korn > 0.01:
            # dreads whip against the motion and fly up when falling; the leash trails behind
            want = max(-1.0, min(1.0, -self.vx / 700)) + self.nod * self.nod_side * 0.6
            self.dread_v += ((want - self.dread_sw) * 60 - self.dread_v * 6) * dt
            self.dread_sw += self.dread_v * dt
            self.leash_x += (self.x - self.leash_x) * min(1, dt * 1.2)
        # desert: sand blowing across the screen on the hot wind
        if self.desert > 0.2 and random.random() < dt * 25 * self.desert:
            self.emit(Particle(
                -10, H * random.uniform(0.35, 1.0), random.uniform(220, 480), random.uniform(-20, 10),
                W / 220, random.uniform(0.07, 0.12), random.uniform(2.0, 4.5), "sand"))
        # Nirvana's quiet verse / loud chorus: shades go up and the eyes go X on the loud parts
        loud = 0.0 if quiet_level(level) else clamp01((a.loud_rel - 0.75) / 0.3)
        self.grunge += (loud - self.grunge) * min(1, dt * (3.0 if loud > self.grunge else 1.2))
        if self.cool > 0.5 and self.ember and random.random() < dt * 7:
            ex, ey = self.ember
            self.particles.append(Particle(
                ex, ey, random.uniform(-8, 12), random.uniform(-55, -35),
                random.uniform(1.6, 2.6), 0.0, random.uniform(2, 4), "smoke"))
        if self.acid > 0.3:
            r0 = self.R
            # the melting bottom drips acid
            if random.random() < dt * (1.5 + 5 * level) * self.acid and len(self.particles) < self.max_particles:
                self.particles.append(Particle(
                    self.x + random.uniform(-0.6, 0.6) * r0, self.y + r0 * 0.95,
                    self.vx * 0.2, random.uniform(0, 40), random.uniform(0.8, 1.3), 0.3,
                    random.uniform(2.5, 4.5), "acid"))
            # and a few black butterflies keep it company
            if random.random() < dt * 0.6 * self.acid:
                ang = random.uniform(0, TAU)
                self.emit(Particle(
                    self.x + math.cos(ang) * r0 * 1.3, self.y + math.sin(ang) * r0 * 1.3,
                    math.cos(ang) * 40, math.sin(ang) * 40 - 30, random.uniform(3.5, 5.5), 0.08,
                    random.uniform(9, 14), "moth"))
        if self.nola > 0.01:
            # the beard swings against the motion and with the headbang
            want = max(-1.0, min(1.0, -self.vx / 900)) + self.nod * self.nod_side * 0.5
            self.beard_v += ((want - self.beard_sw) * 45 - self.beard_v * 5) * dt
            self.beard_sw += self.beard_v * dt
            # a lazy haze of smoke hangs around it
            if self.nola > 0.3 and random.random() < dt * 3 * self.nola and len(self.particles) < self.max_particles:
                ang = random.uniform(0, TAU)
                self.particles.append(Particle(
                    self.x + math.cos(ang) * self.R * 1.1, self.y + math.sin(ang) * self.R * 0.9,
                    random.uniform(-25, 25), random.uniform(-30, -10), random.uniform(2.5, 3.5),
                    0.0, self.R * random.uniform(0.15, 0.25), "smoke"))
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
        if self.desert > 0.01:
            # desert: the body's colour is drawn into sunset reds, oranges and sand
            want = 0.06 + 0.07 * math.sin(self.t * 0.6)
            self.hue += (((want - self.hue) + 0.5) % 1.0 - 0.5) * self.desert * min(1, dt * 3)
        if self.rage > 0.01:
            # RATM: red (and, through dark(), black)
            want = 0.99 + 0.02 * math.sin(self.t * 0.7)
            self.hue += (((want - self.hue) + 0.5) % 1.0 - 0.5) * self.rage * min(1, dt * 3)
        if self.acid > 0.01:
            # Acid Bath: sickly swamp green, drifting toward murky olive
            want = 0.28 + 0.04 * math.sin(self.t * 0.5)
            self.hue += (((want - self.hue) + 0.5) % 1.0 - 0.5) * self.acid * min(1, dt * 3)
        if self.nola > 0.01:
            # Down / COC: whiskey amber and bourbon brown
            want = 0.075 + 0.015 * math.sin(self.t * 0.5)
            self.hue += (((want - self.hue) + 0.5) % 1.0 - 0.5) * self.nola * min(1, dt * 3)
        if self.tool > 0.01:
            # Tool: deep indigo and violet, the colours of the visionary art
            want = 0.72 + 0.06 * math.sin(self.t * 0.4)
            self.hue += (((want - self.hue) + 0.5) % 1.0 - 0.5) * self.tool * min(1, dt * 3)

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
            elif p.kind == "paint":
                # flies off, slows to a stop on the "wall", then just drips
                p.vx *= max(0.0, 1 - 7 * dt)
                p.vy *= max(0.0, 1 - 7 * dt)
            elif p.kind == "sand":
                p.vy += math.sin(self.t * 3 + p.rot) * 60 * dt    # tumbles on the wind
            elif p.kind == "smoke":
                p.vx += math.sin(self.t * 1.5 + p.rot) * 30 * dt   # curls as it rises
                p.size += dt * 9
            elif p.kind == "moth":
                # flutters about in loops, no gravity, slowing after a scatter
                p.vx += math.sin(self.t * 2.3 + p.rot) * 140 * dt
                p.vy += math.cos(self.t * 1.9 + p.rot * 1.3) * 140 * dt - 10 * dt
                p.vx *= max(0.0, 1 - 1.2 * dt)
                p.vy *= max(0.0, 1 - 1.2 * dt)
                p.rot += dt * 2       # (plus the 6 below: wingbeats are drawn from rot)
            elif p.kind != "ring":
                p.vy += 500 * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.rot += dt * 6
        self.particles = [p for p in self.particles if p.life > 0]

    def dark(self, rgb, k=0.85):
        """Fade a colour toward DARK as the music gets heavy."""
        t = max(self.heavy, 0.5 * self.nin, 0.3 * self.rage, 0.35 * self.acid, 0.3 * self.nola) * k   # egg palettes
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
        if self.nin > 0.3 and random.random() < 0.3:
            self.glitch = 1.0
        if self.linkin > 0.3:
            # graffiti: paint splats fly off and stick, dripping
            for _ in range(2):
                ang = random.uniform(0, TAU)
                sp = random.uniform(500, 900)
                self.emit(Particle(
                    self.x + math.cos(ang) * self.R, self.y + math.sin(ang) * self.R,
                    math.cos(ang) * sp, math.sin(ang) * sp, random.uniform(2.5, 3.5),
                    random.choice((0.0, 0.0, 0.58, 0.14)), random.uniform(6, 12), "paint"))
        if self.hover > 0.5:
            # floating in a breakdown: kicks just make it pulse
            self.squash_v += 1.5
            return
        self.jump(strength, W, H, a, synced=True)

    def on_snare(self, strength):
        """Headbang! Body nods down, face dips, little sideways shove."""
        s = min(strength, 3.0)
        self.penta_pulse = 1.0
        if self.nin > 0.3 and random.random() < 0.5:
            self.glitch = 1.0
        if self.rage > 0.5 and self.grunge > 0.4 and self.kill <= 0 and random.random() < 0.25:
            self.kill = 0.35       # Morello's kill-switch stutter
        if self.acid > 0.5 and self.grunge > 0.4:
            # the sludge hits: butterflies burst off the blob (Scream of the Butterfly)
            for _ in range(3):
                ang = random.uniform(0, TAU)
                sp = random.uniform(250, 450)
                self.emit(Particle(
                    self.x + math.cos(ang) * self.R, self.y + math.sin(ang) * self.R,
                    math.cos(ang) * sp, math.sin(ang) * sp, random.uniform(2.5, 4.0), 0.08,
                    random.uniform(9, 14), "moth"))
        self.nod = min(1.0, 0.5 + 0.25 * s + 0.3 * self.nola)   # Down / COC: big slow headbangs
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
        height *= 1 + 0.4 * self.korn        # Korn: jump!
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
            hue = self.hue + random.random()
            if random.random() < self.era90:
                # 90s: Memphis-pattern squiggles and triangles in teal, purple, pink and yellow
                kind = random.choice(("tri", "squig", "squig", "dot"))
                hue = random.choice(MEMPHIS_HUES)
            elif random.random() < self.y2k:
                kind = "glint"                     # early 2000s: chrome four-point sparkles
            self.emit(Particle(
                self.x, self.y, math.cos(ang) * sp, math.sin(ang) * sp,
                random.uniform(0.6, 1.5), hue, random.uniform(4, 11), kind))

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
        sat *= 1 - 0.55 * self.layne      # Layne: washed-out, grungy colours
        sat *= 1 - 0.85 * self.nin        # NIN: nearly black and white
        if self.tool > 0.01:
            self.draw_mandala(cr, a, self.R)   # base size: the bass pulse is built in

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

        if self.nin > 0.3 and self.glitch > 0.05:
            self.draw_glitched(cr, a, r, sat)
        else:
            self.draw_body(cr, a, r, sat)
        if self.korn > 0.01:
            self.draw_leash(cr, r)
        for p in self.particles:
            if p.kind == "smoke":
                self.draw_particle(cr, p)

    def draw_body(self, cr, a, r, sat):
        """The blob itself: glow, body, outline, shine, horns and the rest, and its face."""
        if self.kill > 0 and int(self.t * 22) % 2:
            return     # RATM kill switch: the blob cuts out and back in like the guitar
        cr.save()
        cr.translate(*self.shake)
        # mellow: rocks its head side to side
        tilt = (0.14 * self.mellow + 0.1 * self.nola) * math.sin(self.sway)   # Down / COC: swagger
        if abs(tilt) > 1e-3:
            cr.translate(self.x, self.y)
            cr.rotate(tilt)
            cr.translate(-self.x, -self.y)
        pts = self.outline(a, r)
        if self.desert > 0.01:
            # heat haze: the body shimmers and melts sideways in slow ripples
            d, t = self.desert, self.t
            pts = [(px + math.sin(py * 0.045 + t * 4.5) * r * 0.06 * d,
                    py + math.sin(px * 0.03 - t * 2.0) * r * 0.025 * d) for px, py in pts]
            self.draw_cactus_arms(cr, a, r)
        if self.acid > 0.01:
            # Acid Bath: the bottom half melts into slow, sagging drips
            k, t = self.acid, self.t
            out = []
            for i, (px, py) in enumerate(pts):
                th = i / self.N * TAU
                down = max(0.0, math.sin(th)) ** 2
                out.append((px, py + r * 0.22 * k * down * (0.5 + 0.5 * math.sin(th * 7 + t * 1.5))))
            pts = out
        if self.wright > 0.01:
            self.draw_prism_light(cr, a, r)

        # glow (bigger and brighter while a build-up charges)
        for k, al in ((1.45 + 0.4 * self.charge, 0.06 + 0.12 * self.charge), (1.22, 0.10)):
            cr.save()
            cr.translate(self.x, self.y)
            cr.scale(k, k)
            cr.translate(-self.x, -self.y)
            smooth_closed_path(cr, pts)
            glow = hsv(self.hue + 0.5 * (1 - max(self.desert, self.tool, self.rage, self.acid, self.nola)), sat)  # on-palette
            glow = tuple(lerp(c, d, self.heavy) for c, d in zip(glow, (0.75, 0.0, 0.05)))
            cr.set_source_rgba(*glow, al + a.level * 0.08 + self.crazy * 0.08)
            cr.fill()
            cr.restore()

        # body
        if self.nin > 0.01:
            self.draw_rgb_split(cr, pts, r)
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
            spot = (0.25 + j * 0.13) * (1 - 0.8 * max(self.desert, self.tool, self.rage, self.acid, 1.2 * self.nola))  # egg's hues
            cr.set_source_rgba(*self.dark(hsv(self.hue + spot, 0.9 * sat, 1), 0.9), 0.55)
            cr.arc(self.x + math.cos(ang) * dist, self.y + math.sin(ang) * dist, sr, 0, TAU)
            cr.fill()
        if self.acid > 0.01:
            self.draw_acid_bubbles(cr, a, r)
        if self.nin > 0.01:
            self.draw_scanlines(cr, r)
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
        if self.y2k > 0.01:
            # early 2000s: glossy, candy-like "aqua" sheen across the top
            cr.save()
            cr.translate(self.x, self.y - r * 0.52)
            cr.scale(r * 0.62, r * 0.3)
            cr.arc(0, 0, 1, 0, TAU)
            cr.restore()
            g = cairo.LinearGradient(0, self.y - r * 0.82, 0, self.y - r * 0.25)
            g.add_color_stop_rgba(0, 1, 1, 1, 0.55 * self.y2k)
            g.add_color_stop_rgba(1, 1, 1, 1, 0)
            cr.set_source(g)
            cr.fill()

        if self.horns > 0.02:
            self.draw_horns(cr, r)
        if self.layne > 0.01:
            self.draw_chain(cr, r)
        if self.korn > 0.01:
            self.draw_collar(cr, r)
        if self.acid > 0.01:
            self.draw_moss(cr, r, strands=False)   # the clump sits behind the forehead
        self.draw_face(cr, a, r)
        if self.acid > 0.01:
            self.draw_moss(cr, r, clump=False)     # strands hang down over the sides
        if self.korn > 0.01:
            self.draw_dreads(cr, r)
        cr.restore()

    def draw_glitched(self, cr, a, r, sat):
        """Easter egg (Nine Inch Nails): the blob tears into horizontal slices that jump
        sideways, with static around it. Drawn into a group clipped to the blob's
        surroundings, so the offscreen buffer stays small."""
        x0, y0 = self.x - r * 3.9, self.y - r * 2.1
        w, h = r * 7.8, r * 3.8
        g = self.glitch
        cr.save()
        cr.rectangle(x0, y0, w, h)
        cr.clip()
        cr.push_group()
        self.draw_body(cr, a, r, sat)
        pat = cr.pop_group()
        slices = []
        for _ in range(random.randint(2, 5)):
            sy = self.y + random.uniform(-1.0, 0.9) * r
            sh = r * random.uniform(0.04, 0.22)
            slices.append((sy, sh, random.choice((-1, 1)) * r * random.uniform(0.08, 0.4) * g))
        # everything outside the slices stays put...
        cr.save()
        cr.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        cr.rectangle(x0, y0, w, h)
        for sy, sh, _ in slices:
            cr.rectangle(x0, sy, w, sh)
        cr.clip()
        cr.set_source(pat)
        cr.paint()
        cr.restore()
        # ...and the slices are shifted sideways
        for sy, sh, dx in slices:
            cr.save()
            cr.rectangle(x0, sy, w, sh)
            cr.clip()
            cr.translate(dx, 0)
            cr.set_source(pat)
            cr.paint()
            cr.restore()
        # static
        for _ in range(int(40 * g)):
            v = random.random()
            cr.rectangle(self.x + random.uniform(-1.6, 1.6) * r, self.y + random.uniform(-1.3, 1.3) * r,
                         random.uniform(2, 9), random.uniform(1, 3))
            cr.set_source_rgba(v, v, v, 0.8 * g)
            cr.fill()
        cr.restore()

    def draw_rgb_split(self, cr, pts, r):
        """Easter egg (Nine Inch Nails): red and cyan ghosts of the body, like a misaligned
        CRT; they jump apart during a glitch."""
        off = r * (0.035 + 0.15 * self.glitch) * self.nin
        for dx, rgb in ((off, (1.0, 0.1, 0.15)), (-off, (0.1, 0.9, 1.0))):
            cr.save()
            cr.translate(dx, -dx * 0.15)
            smooth_closed_path(cr, pts)
            cr.set_source_rgba(*rgb, 0.55 * self.nin)
            cr.fill()
            cr.restore()

    def draw_scanlines(self, cr, r):
        """Easter egg (Nine Inch Nails): CRT scanlines across the body (already clipped to it)
        and a brighter band rolling down through it."""
        k = self.nin
        top, bot = self.y - r * 1.4, self.y + r * 1.4
        step = max(3.0, r * 0.045)
        y = top
        while y < bot:
            cr.move_to(self.x - r * 1.5, y)
            cr.line_to(self.x + r * 1.5, y)
            y += step
        cr.set_source_rgba(0, 0, 0, 0.3 * k)
        cr.set_line_width(step * 0.4)
        cr.stroke()
        band = top + (self.t * 0.35 % 1.0) * (bot - top)
        cr.rectangle(self.x - r * 1.5, band, r * 3, r * 0.18)
        cr.set_source_rgba(1, 1, 1, 0.08 * k)
        cr.fill()

    def draw_dreads(self, cr, r):
        """Easter egg (Korn): dreadlocks hanging around the face, whipping with every move."""
        k = self.korn
        sw = self.dread_sw
        lift = max(-0.4, min(0.6, self.vy / 1500))      # fly up while falling, drag while rising
        cr.save()
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        ends = []
        for side in (-1, 1):
            for j in range(4):
                ang = -math.pi / 2 + side * (0.8 + j * 0.22)     # clear of the eyes
                bx = self.x + math.cos(ang) * r * 0.86
                by = self.y + math.sin(ang) * r * 0.86 + self.nod * r * 0.1
                ln = r * (1.0 + 0.12 * j) * k
                # hang down past the side of the face, swinging with the motion
                ex = bx + side * r * (0.12 + 0.05 * j) + sw * r * 0.45
                ey = by + ln * (1 - 0.6 * lift)
                cx1 = bx + side * r * 0.25 + sw * r * 0.1
                cy1 = by + ln * 0.2
                cx2 = ex - sw * r * 0.2
                cy2 = ey - ln * 0.35
                cr.move_to(bx, by)
                cr.curve_to(cx1, cy1, cx2, cy2, ex, ey)
                ends.append((ex, ey))
        path = cr.copy_path()
        cr.set_source_rgba(0.1, 0.06, 0.04, k)
        cr.set_line_width(max(4, r * 0.11))
        cr.stroke()
        cr.translate(-r * 0.015, -r * 0.01)            # highlight (dashes would be slow)
        cr.append_path(path)
        cr.set_source_rgba(0.4, 0.26, 0.15, k)
        cr.set_line_width(max(1.5, r * 0.035))
        cr.stroke()
        cr.translate(r * 0.015, r * 0.01)
        for i, (ex, ey) in enumerate(ends):              # beads at the tips
            cr.arc(ex, ey, max(2.5, r * 0.055), 0, TAU)
            cr.set_source_rgba(*((0.85, 0.65, 0.2) if i % 3 else (0.7, 0.15, 0.15)), k)
            cr.fill()
        cr.restore()

    def draw_collar(self, cr, r):
        """Easter egg (Korn): a black spiked collar low on the body, with a ring for the leash."""
        k = self.korn
        cr.save()
        a0, a1 = math.pi * 0.18, math.pi * 0.82
        rr = r * 0.78
        cr.arc(self.x, self.y - r * 0.05, rr, a0, a1)
        cr.set_source_rgba(0.06, 0.06, 0.07, k)
        cr.set_line_width(r * 0.13)
        cr.stroke()
        # silver spikes pointing outward
        for j in range(7):
            ang = lerp(a0 + 0.12, a1 - 0.12, j / 6)
            c, s_ = math.cos(ang), math.sin(ang)
            bx, by = self.x + c * rr, self.y - r * 0.05 + s_ * rr
            nx, ny = -s_, c
            cr.move_to(bx + nx * r * 0.04, by + ny * r * 0.04)
            cr.line_to(bx + c * r * 0.14, by + s_ * r * 0.14)
            cr.line_to(bx - nx * r * 0.04, by - ny * r * 0.04)
            cr.close_path()
        cr.set_source_rgba(0.85, 0.86, 0.9, k)
        cr.fill()
        # leash ring on the right
        rx, ry = self.leash_ring(r)
        cr.arc(rx, ry, r * 0.07, 0, TAU)
        cr.set_source_rgba(0.8, 0.8, 0.85, k)
        cr.set_line_width(max(2, r * 0.03))
        cr.stroke()
        cr.restore()

    def leash_ring(self, r):
        ang = math.pi * 0.18
        return (self.x + math.cos(ang) * r * 0.86, self.y - r * 0.05 + math.sin(ang) * r * 0.86)

    def draw_leash(self, cr, r):
        """Easter egg (Korn): ...on a leash, held by someone above the top of the screen. It
        goes taut when the blob jumps up toward it and sags when it's far below."""
        k = self.korn
        rx, ry = self.leash_ring(r)
        rx += self.shake[0]
        ry += self.shake[1]
        tx, ty = self.leash_x + r * 0.5, -20.0
        span = max(1.0, ry - ty)
        sag = r * 0.6 * clamp01(1 - span / 900) + r * 0.25    # slack when it's close
        cr.save()
        cr.move_to(rx, ry)
        cr.curve_to(rx + sag, ry - span * 0.3, tx + sag, ty + span * 0.4, tx, ty)
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        cr.set_source_rgba(0.08, 0.08, 0.09, k)
        cr.set_line_width(max(3, r * 0.05))
        cr.stroke_preserve()
        cr.set_source_rgba(0.6, 0.15, 0.15, k)                # red stitching
        cr.set_line_width(max(1, r * 0.015))
        cr.stroke()
        cr.restore()

    def draw_headphones(self, cr, face_y, r):
        """Easter egg (Linkin Park): big DJ headphones; the LEDs on the cups blink on the beat."""
        k = self.linkin
        cy = face_y + r * 0.02
        cr.save()
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        # headband over the top of the head
        cr.arc(self.x, cy, r * 0.98, math.pi + 0.35, TAU - 0.35)
        cr.set_source_rgba(0.08, 0.08, 0.09, k)
        cr.set_line_width(r * 0.11)
        cr.stroke()
        cr.arc(self.x, cy, r * 0.98, math.pi + 0.5, TAU - 0.5)
        cr.set_source_rgba(0.35, 0.35, 0.38, k)
        cr.set_line_width(r * 0.03)
        cr.stroke()
        for side in (-1, 1):
            cx = self.x + side * r * 0.93
            # ear cup
            cr.save()
            cr.translate(cx, cy)
            cr.scale(r * 0.17, r * 0.3)
            cr.arc(0, 0, 1, 0, TAU)
            cr.restore()
            cr.set_source_rgba(0.1, 0.1, 0.11, k)
            cr.fill_preserve()
            cr.set_source_rgba(0.75, 0.05, 0.08, k)          # red rim
            cr.set_line_width(max(2, r * 0.035))
            cr.stroke()
            # LED blinks with the drums
            cr.arc(cx + side * r * 0.04, cy - r * 0.1, r * 0.035, 0, TAU)
            cr.set_source_rgba(1.0, 0.15 + 0.6 * self.penta_pulse, 0.1, k * (0.4 + 0.6 * self.penta_pulse))
            cr.fill()
        cr.restore()

    def draw_scream_waves(self, cr, y, mw, r):
        """Easter egg (Linkin Park): sound waves blasting out of the screaming mouth."""
        k = self.scream
        cr.save()
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        cr.set_line_width(max(2, r * 0.035))
        for i in range(3):
            ph = (self.t * 2.2 + i / 3) % 1.0
            rad = mw * 0.8 + ph * r * 1.1
            for side in (-1, 1):
                mid = 0.0 if side > 0 else math.pi
                cr.new_sub_path()
                cr.arc(self.x, y, rad, mid - 0.5, mid + 0.5)
            cr.set_source_rgba(1, 1, 1, 0.75 * k * (1 - ph))
            cr.stroke()
        cr.restore()

    def draw_cap(self, cr, face_y, r, k):
        """90s: a baseball cap worn backwards, the brim sticking out behind the head."""
        cy = face_y - r * 0.6
        cr.save()
        cr.translate(self.x, cy)
        cr.rotate(-0.12 + 0.05 * math.sin(self.sway))
        # brim, behind the crown
        cr.save()
        cr.translate(r * 0.42, -r * 0.42)
        cr.rotate(-0.45)
        cr.scale(r * 0.42, r * 0.11)
        cr.arc(0, 0, 1, 0, TAU)
        cr.restore()
        cr.set_source_rgba(*hsv(0.78, 0.7, 0.55), k)
        cr.fill()
        # crown
        cr.save()
        cr.scale(1, 0.78)
        cr.arc(0, 0, r * 0.72, math.pi, TAU)
        cr.restore()
        cr.close_path()
        g = cairo.LinearGradient(0, -r * 0.56, 0, 0)
        g.add_color_stop_rgba(0, *hsv(0.48, 0.75, 0.85), k)
        g.add_color_stop_rgba(1, *hsv(0.48, 0.8, 0.6), k)
        cr.set_source(g)
        cr.fill()
        # panel seams and the button on top
        cr.set_source_rgba(0, 0.15, 0.15, 0.6 * k)
        cr.set_line_width(max(1, r * 0.02))
        for sx in (-0.35, 0.0, 0.35):
            cr.move_to(sx * r, 0)
            cr.curve_to(sx * r, -r * 0.3, sx * r * 0.3, -r * 0.5, 0, -r * 0.56)
        cr.stroke()
        cr.arc(0, -r * 0.56, r * 0.06, 0, TAU)
        cr.set_source_rgba(*hsv(0.78, 0.7, 0.55), k)
        cr.fill()
        # backwards: the snapback opening and strap at the front
        cr.arc(0, 0, r * 0.2, math.pi, TAU)
        cr.close_path()
        cr.set_source_rgba(*self.dark(hsv(self.hue, 1, 0.8)), k)
        cr.fill()
        cr.rectangle(-r * 0.2, -r * 0.07, r * 0.4, r * 0.05)
        cr.set_source_rgba(0.1, 0.1, 0.12, k)
        cr.fill()
        cr.move_to(-r * 0.72, 0)
        cr.line_to(r * 0.72, 0)
        cr.set_source_rgba(*hsv(0.48, 0.8, 0.4), k)
        cr.set_line_width(max(2, r * 0.05))
        cr.stroke()
        cr.restore()

    def draw_frosted_tips(self, cr, r, k):
        """Early 2000s: gelled, spiky hair with bleached frosted tips."""
        n = 9
        bases, tips = [], []
        for j in range(n):
            ang = -math.pi / 2 + (j / (n - 1) - 0.5) * 1.55
            wob = 0.04 * math.sin(self.t * 3 + j * 1.7) - 0.05 * self.squash
            a0, a1 = ang - 0.11, ang + 0.11
            b0 = (self.x + math.cos(a0) * r * 0.86, self.y + math.sin(a0) * r * 0.86)
            b1 = (self.x + math.cos(a1) * r * 0.86, self.y + math.sin(a1) * r * 0.86)
            tl = r * (1.22 + 0.06 * (j % 2) + wob)
            tip = (self.x + math.cos(ang + 0.08) * tl, self.y + math.sin(ang + 0.08) * tl)
            bases.append((b0, b1))
            tips.append(tip)
        cr.save()
        for frac, col in ((0.0, (0.35, 0.22, 0.1)), (0.5, (1.0, 0.93, 0.62))):
            # first the whole spikes in brown, then their outer half bleached blonde
            for (b0, b1), tip in zip(bases, tips):
                p0 = (lerp(b0[0], tip[0], frac), lerp(b0[1], tip[1], frac))
                p1 = (lerp(b1[0], tip[0], frac), lerp(b1[1], tip[1], frac))
                cr.move_to(*p0)
                cr.line_to(*tip)
                cr.line_to(*p1)
                cr.close_path()
            cr.set_source_rgba(*col, k)
            cr.fill()
        cr.restore()

    def draw_red_star(self, cr, x, y, size):
        """Easter egg (RATM): a red five-pointed star on the forehead, flaring on the drums."""
        k = self.rage
        size *= 1 + 0.15 * self.penta_pulse
        cr.save()
        for j in range(10):
            ang = -math.pi / 2 + j * math.pi / 5
            rr = size if j % 2 == 0 else size * 0.42
            (cr.move_to if j == 0 else cr.line_to)(x + math.cos(ang) * rr, y + math.sin(ang) * rr)
        cr.close_path()
        cr.set_source_rgba(0.9, 0.05, 0.05, k)
        cr.fill_preserve()
        cr.set_source_rgba(0.0, 0.0, 0.0, k)
        cr.set_line_width(max(2.5, size * 0.2))
        cr.set_line_join(cairo.LINE_JOIN_MITER)
        cr.stroke_preserve()
        cr.set_source_rgba(1.0, 0.1, 0.08, k)                  # repaint the inside over the stroke
        cr.fill()
        cr.restore()

    def draw_bandana(self, cr, face_y, my, mh, r):
        """Easter egg (RATM): a bandana tied over the lower face. It puffs out as the blob
        shouts, and the knot's tails flap at the side."""
        k = self.rage
        top = face_y + r * 0.2
        bot = my + r * 0.42 + mh * 0.5                # bulges with the open mouth
        half = r * 0.88
        cr.save()
        cr.move_to(self.x - half, top + r * 0.05)
        cr.curve_to(self.x - half * 0.4, top - r * 0.06, self.x + half * 0.4, top - r * 0.06,
                    self.x + half, top + r * 0.05)
        cr.curve_to(self.x + half * 0.8, top + r * 0.35, self.x + r * 0.3, bot - r * 0.05,
                    self.x, bot)
        cr.curve_to(self.x - r * 0.3, bot - r * 0.05, self.x - half * 0.8, top + r * 0.35,
                    self.x - half, top + r * 0.05)
        cr.close_path()
        path = cr.copy_path()
        g = cairo.LinearGradient(0, top, 0, bot)
        g.add_color_stop_rgba(0, 0.13, 0.12, 0.13, k)     # black, against the red body
        g.add_color_stop_rgba(1, 0.04, 0.04, 0.05, k)
        cr.set_source(g)
        cr.fill()
        # white paisley-ish dots, clipped to the cloth
        cr.append_path(path)
        cr.clip()
        cr.set_source_rgba(1, 1, 1, 0.75 * k)
        for i in range(-4, 5):
            for j in range(4):
                dx = (i + 0.5 * (j % 2)) * r * 0.2
                dy = top + r * (0.08 + j * 0.15)
                cr.new_sub_path()
                cr.arc(self.x + dx, dy, r * (0.025 if (i + j) % 2 else 0.04), 0, TAU)
        cr.fill()
        cr.reset_clip()
        # fold line and outline
        cr.append_path(path)
        cr.set_source_rgba(0.0, 0.0, 0.0, k)
        cr.set_line_width(max(1.5, r * 0.025))
        cr.stroke()
        # knot and tails on the right, flapping
        kx, ky = self.x + half * 0.95, top + r * 0.08
        flap = math.sin(self.t * 9) * r * 0.06 - self.vx / 3000 * r
        cr.set_source_rgba(0.08, 0.08, 0.09, k)
        for d, ln in ((-1, 0.4), (1, 0.32)):
            cr.move_to(kx, ky - r * 0.05)
            cr.curve_to(kx + r * 0.15, ky + d * r * 0.02, kx + r * ln * 0.7, ky + d * r * 0.06 + flap,
                        kx + r * ln, ky + d * r * 0.1 + flap)
            cr.line_to(kx + r * ln * 0.85, ky + d * r * 0.3 + flap)
            cr.curve_to(kx + r * ln * 0.6, ky + d * r * 0.22 + flap, kx + r * 0.1, ky + d * r * 0.12,
                        kx, ky + r * 0.06)
            cr.close_path()
        cr.fill()
        cr.arc(kx, ky, r * 0.06, 0, TAU)
        cr.fill()
        cr.restore()

    def draw_acid_bubbles(self, cr, a, r):
        """Easter egg (Acid Bath): bubbles rising through the body, faster when it's loud.
        Called with the body as the clip."""
        k = self.acid
        cr.set_line_width(max(1.2, r * 0.02))
        for j in range(9):
            ph = (self.t * (0.25 + 0.05 * (j % 3)) * (1 + a.level) + j / 9) % 1
            bx = self.x + math.sin(j * 2.4 + self.t * 1.3) * r * 0.65 * (1 - 0.3 * ph)
            by = self.y + r * 1.05 - ph * r * 2.0
            br = r * (0.04 + 0.03 * ((j * 7) % 4) / 3) * (0.6 + 0.6 * ph)
            al = k * min(1, ph * 6, (1 - ph) * 6)
            cr.new_sub_path()
            cr.arc(bx, by, br, 0, TAU)
            cr.set_source_rgba(0.75, 1.0, 0.45, 0.25 * al)
            cr.fill_preserve()
            cr.set_source_rgba(0.85, 1.0, 0.6, 0.7 * al)
            cr.stroke()

    def draw_moss(self, cr, r, clump=True, strands=True):
        """Easter egg (Acid Bath): Spanish moss draped over the crown, hanging down the sides
        and swinging with the motion."""
        k = self.acid
        swing = max(-1.0, min(1.0, -self.vx / 900)) * r * 0.12
        cr.save()
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        # the clump on top
        cr.set_source_rgba(0.42, 0.47, 0.36, 0.95 * k * clump)
        for j in range(11):
            ang = -math.pi / 2 + (j - 5) * 0.2
            cr.new_sub_path()
            cr.arc(self.x + math.cos(ang) * r * 0.93, self.y + math.sin(ang) * r * 0.93,
                   r * (0.1 + 0.03 * (j % 3)), 0, TAU)
        cr.fill()
        # strands hanging off both sides of the crown, clear of the eyes
        for j in range(14 * strands):
            side = -1 if j < 7 else 1
            u = (j % 7) / 6
            ang = -math.pi / 2 + side * (0.85 + u * 0.6)
            x0 = self.x + math.cos(ang) * r * 0.92
            y0 = self.y + math.sin(ang) * r * 0.92
            ln = r * (0.35 + 0.25 * ((j * 5) % 3) / 2 + 0.15 * u)
            sw = swing * (0.6 + u) + math.sin(self.t * 1.7 + j) * r * 0.025
            cr.move_to(x0, y0)
            steps = 6
            for s in range(1, steps + 1):
                f = s / steps
                cr.line_to(x0 + side * r * 0.04 * f + sw * f * f + math.sin(f * 9 + j) * r * 0.025,
                           y0 + ln * f)
            g = 0.36 + 0.06 * (j % 3)
            cr.set_source_rgba(g, g + 0.06, g - 0.05, 0.9 * k)
            cr.set_line_width(max(1.5, r * (0.035 - 0.01 * (j % 2))))
            cr.stroke()
        cr.restore()

    def draw_fleur_de_lis(self, cr, x, y, size):
        """Easter egg (Down / COC): New Orleans' gold fleur-de-lis on the forehead, flaring on
        the drums."""
        k = self.nola
        s = size * (1 + 0.15 * self.penta_pulse)
        cr.save()
        cr.translate(x, y)
        cr.scale(s, s)
        # centre petal
        cr.move_to(0, -1.25)
        cr.curve_to(0.45, -0.75, 0.38, -0.05, 0.12, 0.25)
        cr.line_to(-0.12, 0.25)
        cr.curve_to(-0.38, -0.05, -0.45, -0.75, 0, -1.25)
        cr.close_path()
        # side petals, curling out and down
        for d in (-1, 1):
            cr.move_to(d * 0.14, 0.2)
            cr.curve_to(d * 0.2, -0.35, d * 0.75, -0.75, d * 1.0, -0.35)
            cr.curve_to(d * 1.15, -0.05, d * 0.95, 0.25, d * 0.72, 0.2)
            cr.curve_to(d * 0.85, 0.0, d * 0.75, -0.2, d * 0.55, -0.12)
            cr.curve_to(d * 0.38, -0.05, d * 0.32, 0.12, d * 0.3, 0.3)
            cr.close_path()
        # the band
        cr.rectangle(-0.5, 0.25, 1.0, 0.18)
        # the foot below the band
        cr.move_to(-0.12, 0.43)
        cr.curve_to(-0.15, 0.7, -0.35, 0.85, -0.5, 0.9)
        cr.line_to(0.5, 0.9)
        cr.curve_to(0.35, 0.85, 0.15, 0.7, 0.12, 0.43)
        cr.close_path()
        path = cr.copy_path()
        cr.restore()
        cr.save()
        cr.append_path(path)
        cr.set_source_rgba(0.12, 0.06, 0.0, k)
        cr.set_line_width(max(2.5, size * 0.16))
        cr.set_line_join(cairo.LINE_JOIN_ROUND)
        cr.stroke_preserve()
        g = cairo.LinearGradient(x, y - size * 1.2, x, y + size)
        flare = 0.25 * self.penta_pulse
        g.add_color_stop_rgba(0, 1.0, 0.92, 0.55 + flare, k)
        g.add_color_stop_rgba(0.5, 0.95, 0.72 + flare, 0.2 + flare, k)
        g.add_color_stop_rgba(1, 0.7, 0.45, 0.08, k)
        cr.set_source(g)
        cr.fill()
        cr.restore()

    def draw_beard(self, cr, my, mw, mh, r):
        """Easter egg (Down / COC): a big bushy Southern beard and moustache. The jaw drops it
        as the mouth opens, and it swings with the motion and the headbang."""
        k = self.nola
        sw = self.beard_sw * r * 0.25
        top = my + mh + r * 0.14                # hangs from under the lower lip, drops with the jaw
        ln = r * (0.65 + 0.1 * self.nod)
        dark, mid = (0.2, 0.11, 0.05), (0.33, 0.19, 0.08)
        cr.save()
        # fluffy mass: overlapping tufts, narrowing to the tip
        cr.set_source_rgba(*dark, 0.97 * k)
        for i in range(7):
            f = i / 6
            w = r * (0.62 - 0.42 * f)
            cy = top + ln * f * 0.85
            cx = self.x + sw * f * f
            for j in range(-2, 3):
                cr.new_sub_path()
                cr.arc(cx + j * w * 0.45, cy + abs(j) * r * -0.04, r * (0.17 - 0.06 * f), 0, TAU)
        # sideburns up the cheeks around the mouth
        for d in (-1, 1):
            cr.new_sub_path()
            cr.arc(self.x + d * (mw + r * 0.14), my + r * 0.12, r * 0.15, 0, TAU)
            cr.new_sub_path()
            cr.arc(self.x + d * (mw + r * 0.18), my - r * 0.08, r * 0.12, 0, TAU)
        cr.fill()
        # strands for texture
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        cr.set_line_width(max(1.2, r * 0.02))
        cr.set_source_rgba(*mid, 0.9 * k)
        for i in range(11):
            u = i / 10 * 2 - 1
            x0 = self.x + u * r * 0.5
            y0 = top + r * 0.05 + abs(u) * r * 0.05
            l = ln * (0.85 - 0.45 * abs(u))
            cr.move_to(x0, y0)
            cr.curve_to(x0 + u * r * 0.05, y0 + l * 0.4, x0 + sw * 0.6 - u * r * 0.1, y0 + l * 0.75,
                        x0 + sw - u * r * 0.25, y0 + l)
        cr.stroke()
        # moustache over the top lip, drooping at the ends
        cr.set_source_rgba(*dark, 0.97 * k)
        for d in (-1, 1):
            cr.move_to(self.x, my - r * 0.1)
            cr.curve_to(self.x + d * mw * 0.6, my - r * 0.17, self.x + d * mw * 1.1, my - r * 0.08,
                        self.x + d * (mw + r * 0.08), my + r * 0.18)
            cr.curve_to(self.x + d * mw * 0.9, my + r * 0.02, self.x + d * mw * 0.5, my - r * 0.02,
                        self.x, my + r * 0.0)
            cr.close_path()
        cr.fill()
        cr.restore()

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
            # inner/outer base corners mirror with the side, so both horns are symmetric
            inner, outer = bx - side * w, bx + side * w
            cr.move_to(inner, by + w * 0.3)
            cr.curve_to(inner + side * length * 0.1, by - length * 0.5,
                        tx - side * length * 0.25, ty + length * 0.15, tx, ty)
            cr.curve_to(tx - side * length * 0.05, ty + length * 0.35,
                        outer + side * length * 0.25, by - length * 0.4, outer, by + w * 0.3)
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
        # Kurt: X eyes in the loud parts (shut lids with an X drawn over them)
        xeyes = clamp01((self.kurt * self.grunge - 0.35) / 0.3) * (1 - self.cool)
        lid = max(lid, xeyes, 0.55 * self.scream)    # squeezes its eyes while screaming

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

        drawn_eyes = []
        for i, ((ex, ey, es), p) in enumerate(zip(eyes, self.pupils)):
            grow = 1 + 0.3 * C * (1 + 0.3 * math.sin(self.t * 20 + i * 2))
            ex_ = self.x + ex * r
            ey_ = face_y + ey * r
            er = r * 0.27 * es * grow
            drawn_eyes.append((ex_, ey_, er))
            self.draw_eye(cr, ex_, ey_, er, p, lid, a, idx=i)
            if xeyes > 0.02:
                d = er * 0.6
                cr.save()
                cr.set_line_cap(cairo.LINE_CAP_ROUND)
                cr.set_line_width(max(3, er * 0.22))
                cr.set_source_rgba(0.1, 0, 0.1, xeyes)
                cr.move_to(ex_ - d, ey_ - d)
                cr.line_to(ex_ + d, ey_ + d)
                cr.move_to(ex_ + d, ey_ - d)
                cr.line_to(ex_ - d, ey_ + d)
                cr.stroke()
                cr.restore()
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
        # (Tool: the third eye stays wide open, even through the heaviest riffs)
        third = max(self.third_eye * clamp01(1 - self.heavy * 2), self.tool)
        if third > 0.01:
            s = min(1, third * 2)
            if self.tool > 0.01:
                self.draw_seed_of_life(cr, self.x, face_y - r * 0.5, r * 0.16 * s, self.tool)
            self.draw_eye(cr, self.x, face_y - r * 0.5, r * 0.16 * s, self.pupils[0], 0, a,
                          hue_eye=True, idx=2)
        if self.heavy > 0.15 and self.tool < 0.5 and self.rage < 0.5 and self.nola < 0.5:
            self.draw_pentagram(cr, self.x, face_y - r * 0.7, r * 0.2)
        elif self.nola > 0.01:
            self.draw_fleur_de_lis(cr, self.x, face_y - r * 0.7, r * 0.22)
        elif self.rage > 0.01:
            self.draw_red_star(cr, self.x, face_y - r * 0.68, r * 0.2)
        elif self.wright > 0.01:
            self.draw_mini_prism(cr, self.x, face_y - r * 0.68, r * 0.17)

        # mouth: opens with volume; small & lopsided when lazy, huge grin when crazy
        # mellow: a wide, gentle, mostly closed smile instead of the lazy "meh"
        mw = r * 0.42 * lerp(0.6, 1.25, m) * (1 - yawn_open * 0.45) * (1 + 0.3 * M)
        mh = r * (0.04 + min(a.level, 1.2) * 0.32) * lerp(0.35, 1.5, m) * (1 - sleepy) * (1 - 0.5 * M)
        mh = max(mh, r * 0.55 * yawn_open)
        sc = self.scream                          # Linkin Park: SCREAMS the chorus
        mh = lerp(mh, r * (0.5 + 0.12 * min(a.level, 1.2)), sc)
        mw *= 1 - 0.25 * sc
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

        if xeyes > 0.02:
            self.draw_tongue(cr, self.x + mw * 0.3, my + mh * 0.5, r, xeyes)
        if self.scream > 0.05:
            self.draw_scream_waves(cr, my + mh * 0.3, mw, r)
        if self.linkin > 0.01:
            self.draw_headphones(cr, face_y, r)
        # era headwear, unless an easter egg already has something on (or growing out of) the head
        bare = 1 - max(self.korn, self.linkin, self.tool, self.horns, self.wright,
                       self.kurt * self.grunge, self.evil, self.rage, self.acid, self.nola)
        if self.era90 * bare > 0.01:
            self.draw_cap(cr, face_y, r, self.era90 * bare)
        if self.y2k * bare > 0.01:
            self.draw_frosted_tips(cr, r, self.y2k * bare)
        if self.rage > 0.01:
            self.draw_bandana(cr, face_y, my, mh, r)
        if self.nola > 0.01:
            self.draw_beard(cr, my, mw, mh, r)
        if self.layne > 0.01 and self.cool < 0.99:
            self.draw_round_glasses(cr, drawn_eyes, r, self.layne * (1 - self.cool))
        if self.kurt > 0.01 and self.cool < 0.99:
            self.draw_white_shades(cr, face_y, r, self.kurt * (1 - self.cool))
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

    def draw_chain(self, cr, r):
        """Easter egg (Layne Staley): a chain hung around the blob, padlock at the bottom.
        It swings as the blob moves and jumps on the headbang."""
        k = self.layne
        swing = max(-1.0, min(1.0, -self.vx / 900)) * r * 0.25 + math.sin(self.sway) * r * 0.08
        x0, y0 = self.x - r * 0.9, self.y + r * 0.3
        x2, y2 = self.x + r * 0.9, self.y + r * 0.3
        x1, y1 = self.x + swing * 2, self.y + r * (1.25 - 0.25 * self.nod)  # control point

        def at(u):
            v = 1 - u
            return (v * v * x0 + 2 * v * u * x1 + u * u * x2,
                    v * v * y0 + 2 * v * u * y1 + u * u * y2)

        n = 15
        lw = max(1.5, r * 0.035)
        cr.save()
        cr.set_line_width(lw)
        for i in range(n):
            u = (i + 0.5) / n
            (ax, ay), (bx, by) = at(u - 0.5 / n), at(u + 0.5 / n)
            cx, cy = (ax + bx) / 2, (ay + by) / 2
            ang = math.atan2(by - ay, bx - ax)
            ln = math.hypot(bx - ax, by - ay) * 0.75
            cr.save()
            cr.translate(cx, cy)
            cr.rotate(ang)
            if i % 2 == 0:
                cr.scale(ln, ln * 0.55)            # flat link, seen from the front
                cr.arc(0, 0, 1, 0, TAU)
                cr.restore()
            else:
                cr.move_to(-ln, 0)                 # link turned edge-on
                cr.line_to(ln, 0)
                cr.restore()
            cr.set_source_rgba(0.12, 0.12, 0.14, 0.9 * k)
            cr.set_line_width(lw * 2.2)
            cr.stroke_preserve()
            cr.set_source_rgba(0.78, 0.8, 0.84, k)
            cr.set_line_width(lw)
            cr.stroke()
        # padlock dangling from the lowest point
        px, py = at(0.5)
        s = r * 0.13
        cr.translate(px, py + s * 0.4)
        cr.rotate(0.15 * math.sin(self.t * 2.3) + swing / r * 0.6)
        cr.arc_negative(0, s * 0.35, s * 0.42, 0, math.pi)          # shackle
        cr.set_source_rgba(0.75, 0.77, 0.8, k)
        cr.set_line_width(max(2, s * 0.22))
        cr.stroke()
        cr.rectangle(-s * 0.65, s * 0.35, s * 1.3, s)                # body
        g = cairo.LinearGradient(-s, 0, s, s)
        g.add_color_stop_rgba(0, 0.85, 0.72, 0.35, k)
        g.add_color_stop_rgba(1, 0.5, 0.38, 0.12, k)
        cr.set_source(g)
        cr.fill_preserve()
        cr.set_source_rgba(0.2, 0.15, 0.05, k)
        cr.set_line_width(max(1.5, s * 0.1))
        cr.stroke()
        cr.arc(0, s * 0.75, s * 0.13, 0, TAU)                        # keyhole
        cr.move_to(0, s * 0.75)
        cr.line_to(0, s * 1.1)
        cr.stroke()
        cr.restore()

    def draw_round_glasses(self, cr, eyes, r, k):
        """Easter egg (Layne Staley): little round wire glasses with red lenses."""
        cr.save()
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        lw = max(1.5, r * 0.035)
        lens = [(x, y, er * 1.08) for x, y, er in eyes]
        for x, y, lr in lens:
            cr.arc(x, y, lr, 0, TAU)
            cr.set_source_rgba(0.85, 0.05, 0.08, 0.45 * k)
            cr.fill_preserve()
            cr.set_source_rgba(0.15, 0.15, 0.17, k)
            cr.set_line_width(lw)
            cr.stroke()
            # glint
            cr.arc(x, y, lr * 0.7, math.pi * 1.1, math.pi * 1.45)
            cr.set_source_rgba(1, 1, 1, 0.5 * k)
            cr.set_line_width(lw * 1.2)
            cr.stroke()
        (x0, y0, r0), (x1, y1, r1) = lens
        cr.set_source_rgba(0.15, 0.15, 0.17, k)
        cr.set_line_width(lw)
        cr.move_to(x0 + r0, y0 - r0 * 0.2)                           # bridge
        cr.curve_to(x0 + r0 + r * 0.03, y0 - r0 * 0.5, x1 - r1 - r * 0.03, y1 - r1 * 0.5,
                    x1 - r1, y1 - r1 * 0.2)
        cr.move_to(x0 - r0, y0 - r0 * 0.2)                           # arms
        cr.line_to(self.x - r * 0.93, y0 - r0 * 0.45)
        cr.move_to(x1 + r1, y1 - r1 * 0.2)
        cr.line_to(self.x + r * 0.93, y1 - r1 * 0.45)
        cr.stroke()
        cr.restore()

    def draw_white_shades(self, cr, face_y, r, k):
        """Easter egg (Kurt Cobain): big white oval sunglasses; pushed up onto the forehead
        while the chorus is loud."""
        up = self.grunge * self.grunge * (3 - 2 * self.grunge)       # smoothstep
        y = lerp(face_y - r * 0.04, face_y - r * 0.78, up) - (1 - k) ** 2 * r * 1.2
        rx, ry = r * 0.31, r * 0.24
        cr.save()
        cr.translate(self.x, y)
        cr.rotate(-0.08 * up)                                         # a bit crooked up there
        cr.set_line_join(cairo.LINE_JOIN_ROUND)
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        cr.set_source_rgba(0.96, 0.96, 0.94, k)
        cr.set_line_width(max(2.5, r * 0.06))
        cr.move_to(-r * 0.08, -ry * 0.3)                             # bridge
        cr.curve_to(-r * 0.03, -ry * 0.6, r * 0.03, -ry * 0.6, r * 0.08, -ry * 0.3)
        cr.move_to(-r * 0.36 - rx, -ry * 0.2)                        # arms
        cr.line_to(-r * 0.92, -ry * 0.5)
        cr.move_to(r * 0.36 + rx, -ry * 0.2)
        cr.line_to(r * 0.92, -ry * 0.5)
        cr.stroke()
        for side in (-1, 1):
            cx = side * r * 0.37
            cr.save()
            cr.translate(cx, 0)
            cr.rotate(side * 0.12)                                   # cat-eye tilt
            cr.scale(rx, ry)
            cr.arc(0, 0, 1, 0, TAU)
            cr.restore()
            g = cairo.LinearGradient(cx, -ry, cx, ry)
            g.add_color_stop_rgba(0, 0.2, 0.2, 0.24, k)
            g.add_color_stop_rgba(1, 0.02, 0.02, 0.03, k)
            cr.set_source(g)
            cr.fill_preserve()
            cr.set_source_rgba(0.98, 0.98, 0.96, k)                   # thick white frame
            cr.set_line_width(max(3, r * 0.08))
            cr.stroke_preserve()
            cr.set_source_rgba(0.55, 0.55, 0.52, 0.6 * k)
            cr.set_line_width(1)
            cr.stroke()
            cr.move_to(cx - rx * 0.55, -ry * 0.3)                     # glint
            cr.line_to(cx - rx * 0.1, -ry * 0.45)
            cr.line_to(cx - rx * 0.35, ry * 0.15)
            cr.close_path()
            cr.set_source_rgba(1, 1, 1, 0.3 * k)
            cr.fill()
        cr.restore()

    def draw_tongue(self, cr, x, y, r, k):
        """Easter egg (Kurt Cobain): tongue hanging out with the X eyes."""
        w = r * 0.2
        ln = r * 0.36 * k
        wig = math.sin(self.t * 6) * w * 0.25
        cr.save()
        cr.move_to(x - w * 0.5, y)
        cr.line_to(x - w * 0.5 + wig * 0.3, y + ln)
        cr.arc_negative(x + wig * 0.3, y + ln, w * 0.5, math.pi, 0)
        cr.line_to(x + w * 0.5, y)
        cr.close_path()
        cr.set_source_rgba(1.0, 0.42, 0.55, k)
        cr.fill_preserve()
        cr.set_source_rgba(0.45, 0.05, 0.15, 0.8 * k)
        cr.set_line_width(max(1.5, r * 0.025))
        cr.stroke()
        cr.move_to(x, y + ln * 0.15)                                  # middle groove
        cr.line_to(x + wig * 0.2, y + ln * 0.75)
        cr.stroke()
        cr.restore()

    def draw_prism_light(self, cr, a, r):
        """Easter egg (Richard Wright): a white beam enters the blob from the left and comes out
        the right as a rainbow, each colour band brightening with its part of the spectrum.
        Drawn before the body, so the blob itself is the prism."""
        k = self.wright
        cr.save()
        # incoming white beam
        bx0, by0 = self.x - r * 3.4, self.y + r * 0.55
        bx1, by1 = self.x - r * 0.3, self.y - r * 0.05
        g = cairo.LinearGradient(bx0, by0, bx1, by1)
        g.add_color_stop_rgba(0, 1, 1, 1, 0)
        g.add_color_stop_rgba(0.5, 1, 1, 1, 0.8 * k)
        g.add_color_stop_rgba(1, 1, 1, 1, 0.9 * k)
        cr.set_source(g)
        cr.set_line_width(max(2, r * 0.05))
        cr.set_line_cap(cairo.LINE_CAP_BUTT)
        cr.move_to(bx0, by0)
        cr.line_to(bx1, by1)
        cr.stroke()
        # outgoing rainbow, fanning out and fading
        colors = ((1.0, 0.1, 0.1), (1.0, 0.55, 0.05), (1.0, 0.95, 0.1),
                  (0.15, 0.85, 0.25), (0.15, 0.45, 1.0), (0.6, 0.2, 0.95))
        ex, ey = self.x + r * 0.3, self.y + r * 0.0
        fx, fy = self.x + r * 3.6, self.y + r * 0.55
        w0, w1 = r * 0.035, r * 0.17
        n = len(colors)
        for i, c in enumerate(colors):
            lvl = clamp01(a.spectrum[min(31, 2 + i * 5)])
            al = k * (0.5 + 0.5 * lvl)
            o0, o1 = (i - n / 2) * w0, (i - n / 2) * w1
            cr.move_to(ex, ey + o0)
            cr.line_to(fx, fy + o1)
            cr.line_to(fx, fy + o1 + w1)
            cr.line_to(ex, ey + o0 + w0)
            cr.close_path()
            g = cairo.LinearGradient(ex, ey, fx, fy)
            g.add_color_stop_rgba(0, *c, al)
            g.add_color_stop_rgba(0.7, *c, al * 0.7)
            g.add_color_stop_rgba(1, *c, 0)
            cr.set_source(g)
            cr.fill()
        cr.restore()

    def draw_cactus_arms(self, cr, a, r):
        """Easter egg (John Garcia): saguaro arms sprout from the sides and wave to the riff."""
        k = self.desert
        cr.save()
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        cr.set_line_join(cairo.LINE_JOIN_ROUND)
        w = r * 0.24
        for side, ph in ((-1, 0.0), (1, 1.7)):
            wave = math.sin(self.sway + ph) * 0.5 + 0.5
            out = r * 0.65 * k
            up = r * (0.45 + 0.25 * wave + 0.2 * min(a.bass, 1.0)) * k
            bx, by = self.x + side * r * 0.6, self.y + r * (0.25 - 0.1 * side)
            ex = bx + side * out
            cr.move_to(bx, by)
            cr.line_to(ex - side * w * 0.5, by)
            cr.curve_to(ex, by, ex, by, ex, by - w * 0.5)
            cr.line_to(ex + side * wave * r * 0.06, by - up)
            path = cr.copy_path()
            cr.set_source_rgba(0.08, 0.25, 0.1, k)            # outline
            cr.set_line_width(w + max(3, r * 0.05))
            cr.stroke()
            cr.append_path(path)
            cr.set_source_rgba(0.25, 0.62, 0.28, k)           # flesh
            cr.set_line_width(w)
            cr.stroke()
            cr.append_path(path)
            cr.set_source_rgba(0.45, 0.8, 0.42, 0.7 * k)      # rib highlight
            cr.set_line_width(max(1.5, w * 0.15))
            cr.stroke()
            # spines along the upright part
            tx = ex + side * wave * r * 0.06
            cr.set_source_rgba(1.0, 0.95, 0.75, k)
            cr.set_line_width(max(1, r * 0.012))
            for j in range(5):
                u = (j + 0.5) / 5
                sx, sy = lerp(ex, tx, u), lerp(by - w * 0.5, by - up, u)
                for d in (-1, 1):
                    cr.move_to(sx + d * w * 0.5, sy)
                    cr.line_to(sx + d * w * 0.8, sy - w * 0.15)
            cr.stroke()
        cr.restore()

    def draw_desert_sun(self, cr, W, H, a):
        """Easter egg (John Garcia): a huge striped sunset behind everything, rippling in
        the heat haze and swelling with the bass."""
        k = self.desert
        if k < 0.01:
            return
        R = min(W, H) * 0.22 * (1 + 0.06 * min(a.bass, 1.5))
        cx, cy = W * 0.5, H * 0.62
        top = (1.0, 0.92, 0.35)
        mid = (1.0, 0.45, 0.12)
        bot = (0.85, 0.1, 0.45)
        n = 40
        h = 2 * R / n
        scroll = (self.t * 0.25) % 1.0
        for i in range(n):
            y = cy - R + i * h
            u = (i + 0.5) / n
            # retro sunset cuts in the lower half, drifting down and widening
            if u > 0.5:
                band = ((u - 0.5) * 8 - scroll) % 1.0
                if band < 0.15 + 0.5 * (u - 0.5):
                    continue
            dy = abs(y + h / 2 - cy)
            half = math.sqrt(max(0.0, R * R - dy * dy))
            shimmer = math.sin(y * 0.06 + self.t * 4) * R * 0.03 * u
            c = (tuple(lerp(p, q, u * 2) for p, q in zip(top, mid)) if u < 0.5 else
                 tuple(lerp(p, q, u * 2 - 1) for p, q in zip(mid, bot)))
            cr.set_source_rgba(*c, 0.55 * k)
            cr.rectangle(cx - half + shimmer, y, 2 * half, h + 0.5)
            cr.fill()

    def draw_mandala(self, cr, a, r):
        """Easter egg (Tool): a fractal mandala of hexagrams spinning behind the blob (every
        hexagon spawns smaller, counter-rotating ones at its corners), wrapped in a golden
        spiral. It breathes with the bass and flares on the drums.
        Hundreds of overlapping strokes are too slow to draw every frame, so it's drawn into an
        offscreen image that is pasted each frame. The next image is drawn in the background a
        third at a time (big shapes, small hexagons, spiral), then swapped in, so no single
        frame pays for all of it."""
        half = int(r * 3.4) + 4            # fits the mandala and spiral at their biggest
        if self.mandala is None or self.mandala["half"] != half:
            self.mandala = {"half": half, "stage": 0, "shown": False,
                            "bufs": [cairo.ImageSurface(cairo.FORMAT_ARGB32, 2 * half, 2 * half)
                                     for _ in range(2)]}
        m = self.mandala
        back = m["bufs"][1]
        c = cairo.Context(back)
        c.set_antialias(cairo.ANTIALIAS_FAST)
        c.translate(half, half)
        if m["stage"] == 0:
            c.save()
            c.set_operator(cairo.OPERATOR_CLEAR)
            c.paint()
            c.restore()
            m["geo"] = self.mandala_geometry(a, r)
        self.render_mandala(c, r, m["geo"], m["stage"])
        m["stage"] = (m["stage"] + 1) % 3
        if m["stage"] == 0:
            m["bufs"].reverse()
            m["shown"] = True
        if m["shown"]:
            cr.set_source_surface(m["bufs"][0], self.x - half, self.y - half)
            cr.paint()

    def mandala_geometry(self, a, r):
        """Where every hexagon of the mandala goes, plus what the rest of the drawing needs."""
        t = self.anim_t
        pulse = 1 + 0.08 * min(a.bass, 1.5) + 0.12 * self.penta_pulse
        levels = [[] for _ in range(3)]

        def grow(cx, cy, size, ang, depth):
            levels[depth].append((cx, cy, size, ang))
            if depth < 2:
                for j in range(6):
                    va = ang + j * TAU / 6
                    grow(cx + math.cos(va) * size, cy + math.sin(va) * size, size * 0.42,
                         -ang * 1.7 + t * 0.2, depth + 1)

        grow(0.0, 0.0, r * 1.6 * pulse, t * 0.15, 0)
        return {"levels": levels, "t": t, "pulse": pulse, "k": self.tool, "flare": self.penta_pulse}

    def render_mandala(self, cr, r, geo, stage):
        """Draw one stage of the mandala centred on (0, 0): 0 = the two big levels of hexagons
        with hexagrams inside, 1 = the small hexagons, 2 = the golden spiral."""
        k, t = geo["k"], geo["t"]
        cr.set_line_join(cairo.LINE_JOIN_ROUND)
        if stage < 2:
            for depth in ((0, 1) if stage == 0 else (2,)):
                for cx, cy, size, ang in geo["levels"][depth]:
                    # hexagon, with a hexagram (two triangles) inside on the big ones
                    for j in range(6):
                        va = ang + j * TAU / 6
                        (cr.move_to if j == 0 else cr.line_to)(cx + math.cos(va) * size,
                                                               cy + math.sin(va) * size)
                    cr.close_path()
                    for off in ((0, 1) if depth < 2 else ()):
                        for j in range(3):
                            va = ang + (2 * j + off) * TAU / 6
                            (cr.move_to if j == 0 else cr.line_to)(cx + math.cos(va) * size,
                                                                   cy + math.sin(va) * size)
                        cr.close_path()
                col = hsv(0.12 + depth * 0.4 + 0.04 * math.sin(t * 0.3), 0.8, 1)  # gold, cyan, magenta
                cr.set_source_rgba(*col, k * (0.6 - 0.12 * depth + 0.3 * geo["flare"]))
                cr.set_line_width(max(1, r * 0.015) / (depth * 0.5 + 1))
                cr.stroke()
            return
        # golden spiral (r = a * phi^(2θ/π)), two arms, slowly turning the other way
        b = math.log((1 + 5 ** 0.5) / 2) / (math.pi / 2)
        for arm in (0, math.pi):
            for i in range(121):
                th = i / 120 * 4 * math.pi
                rr = r * 0.055 * math.exp(b * th) * geo["pulse"]
                ang = th + arm - t * 0.3
                (cr.move_to if i == 0 else cr.line_to)(math.cos(ang) * rr, math.sin(ang) * rr)
        cr.set_source_rgba(1.0, 0.82, 0.35, 0.6 * k)
        cr.set_line_width(max(1.5, r * 0.025))
        cr.stroke()

    def draw_seed_of_life(self, cr, x, y, er, k):
        """Easter egg (Tool): six overlapping circles in gold around the third eye."""
        cr.save()
        cr.set_line_width(max(1, er * 0.08))
        for j in range(6):
            va = j * TAU / 6 + self.t * 0.3
            cr.new_sub_path()
            cr.arc(x + math.cos(va) * er * 1.1, y + math.sin(va) * er * 1.1, er * 1.1, 0, TAU)
        cr.new_sub_path()
        cr.arc(x, y, er * 2.2, 0, TAU)
        cr.set_source_rgba(1.0, 0.82, 0.35, 0.8 * k)
        cr.stroke()
        cr.restore()

    def draw_mini_prism(self, cr, x, y, size):
        """Easter egg (Richard Wright): a little glowing prism on the forehead."""
        k = self.wright
        pts = [(x + math.cos(-math.pi / 2 + j * TAU / 3) * size,
                y + math.sin(-math.pi / 2 + j * TAU / 3) * size) for j in range(3)]
        cr.save()
        cr.set_line_join(cairo.LINE_JOIN_ROUND)
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        lw = max(1.2, size * 0.1)
        # beam in
        cr.move_to(x - size * 2.0, y + size * 0.45)
        cr.line_to(x - size * 0.45, y + size * 0.05)
        cr.set_source_rgba(1, 1, 1, 0.9 * k)
        cr.set_line_width(lw)
        cr.stroke()
        # rainbow out
        for i, c in enumerate(((1, 0.1, 0.1), (1, 0.6, 0.05), (1, 0.95, 0.1),
                               (0.15, 0.85, 0.25), (0.2, 0.5, 1), (0.6, 0.2, 0.95))):
            cr.move_to(x + size * 0.4, y + size * 0.1 + i * lw * 0.15)
            cr.line_to(x + size * 2.0, y + size * 0.35 + i * lw * 0.95)
            cr.set_source_rgba(*c, k)
            cr.set_line_width(lw * 0.9)
            cr.stroke()
        # the prism itself
        cr.move_to(*pts[0])
        for p in pts[1:]:
            cr.line_to(*p)
        cr.close_path()
        cr.set_source_rgba(0.05, 0.05, 0.08, 0.85 * k)
        cr.fill_preserve()
        cr.set_source_rgba(1, 1, 1, 0.95 * k)
        cr.set_line_width(lw)
        cr.stroke()
        cr.restore()

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
        if sp < 0.98 and self.desert > 0.02 and not hue_eye:
            # desert trip: pulsing rainbow rings around the pupils
            for j in range(4, 0, -1):
                cr.arc(px, py, pr * (1 + 0.22 * j + 0.1 * math.sin(self.t * 5 - j)), 0, TAU)
                cr.set_source_rgba(*hsv(self.t * 0.4 + j * 0.17 + idx * 0.3, 0.9, 1),
                                   self.desert * (1 - sp))
                cr.fill()
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
        if p.kind == "glint":
            # early 2000s: a silvery four-point sparkle with a hint of colour
            s = p.size * (0.6 + 0.6 * f)
            cr.save()
            cr.translate(p.x, p.y)
            cr.rotate(p.rot * 0.3)
            cr.move_to(0, -s * 1.8)
            for j in range(4):
                a0 = -math.pi / 2 + j * math.pi / 2
                cr.line_to(math.cos(a0 + math.pi / 4) * s * 0.3, math.sin(a0 + math.pi / 4) * s * 0.3)
                cr.line_to(math.cos(a0 + math.pi / 2) * s * 1.8, math.sin(a0 + math.pi / 2) * s * 1.8)
            cr.close_path()
            cr.set_source_rgba(*hsv(p.hue, 0.25, 1), min(1, f * 1.5))
            cr.fill()
            cr.restore()
            return
        if p.kind == "paint":
            # graffiti splat: blob, a few satellite drops, and a drip that runs down over time
            al = min(1.0, f * 3)
            col = (0.05, 0.05, 0.06) if p.hue == 0.58 else hsv(p.hue, 0.95, 0.95)
            if p.hue == 0.14:
                col = (0.96, 0.96, 0.94)                     # white
            cr.set_source_rgba(*col, al)
            s = p.size
            cr.arc(p.x, p.y, s, 0, TAU)
            cr.new_sub_path()
            for j in range(3):
                ang = p.max * 40 + j * 2.1                  # fixed per splat (rot spins)
                cr.arc(p.x + math.cos(ang) * s * 1.7, p.y + math.sin(ang) * s * 1.7, s * 0.3, 0, TAU)
                cr.new_sub_path()
            drip = s * 4 * (1 - f) if math.hypot(p.vx, p.vy) < 40 else 0.0
            if drip > 1:
                cr.rectangle(p.x - s * 0.25, p.y, s * 0.5, drip)
                cr.new_sub_path()
                cr.arc(p.x, p.y + drip, s * 0.3, 0, TAU)
            cr.fill()
            return
        if p.kind == "acid":
            # glowing green drop, stretched by its fall speed
            al = min(1, f * 2)
            s = p.size
            stretch = 1 + min(p.vy / 500, 1.8)
            cr.save()
            cr.translate(p.x, p.y)
            cr.scale(1, stretch)
            cr.arc(0, 0, s * 1.8, 0, TAU)
            cr.set_source_rgba(0.55, 1.0, 0.1, 0.18 * al)
            cr.fill()
            cr.move_to(0, -s * 1.8)
            cr.curve_to(s * 0.4, -s * 0.8, s, -s * 0.4, s, 0)
            cr.arc(0, 0, s, 0, math.pi)
            cr.curve_to(-s, -s * 0.4, -s * 0.4, -s * 0.8, 0, -s * 1.8)
            cr.set_source_rgba(0.62, 1.0, 0.15, al)
            cr.fill()
            cr.restore()
            return
        if p.kind == "moth":
            # black butterfly, wings beating (seen edge-on at the bottom of each stroke)
            al = min(1, f * 3, (1 - f) * 6)
            s = p.size
            beat = 0.25 + 0.75 * abs(math.sin(p.rot * 3))
            cr.save()
            cr.translate(p.x, p.y)
            cr.rotate(math.atan2(p.vy, p.vx) * 0.25 + 0.3 * math.sin(p.rot))
            for side in (-1, 1):
                cr.save()
                cr.scale(side * beat, 1)
                cr.move_to(0, 0)
                cr.curve_to(s * 0.6, -s * 1.6, s * 1.9, -s * 1.3, s * 1.4, -s * 0.1)   # forewing
                cr.curve_to(s * 1.6, s * 0.6, s * 0.9, s * 1.3, 0, s * 0.3)            # hindwing
                cr.close_path()
                cr.set_source_rgba(0.05, 0.04, 0.05, 0.95 * al)
                cr.fill_preserve()
                cr.set_source_rgba(0.75, 0.55, 0.3, 0.6 * al)                          # pale edge
                cr.set_line_width(max(0.8, s * 0.1))
                cr.stroke()
                cr.arc(s * 1.05, -s * 0.75, s * 0.18, 0, TAU)                          # eyespot
                cr.set_source_rgba(0.85, 0.75, 0.55, 0.8 * al)
                cr.fill()
                cr.restore()
            cr.set_source_rgba(0.02, 0.02, 0.02, al)
            cr.save()
            cr.scale(0.3, 1)
            cr.arc(0, s * 0.1, s * 0.6, 0, TAU)
            cr.restore()
            cr.fill()
            cr.restore()
            return
        if p.kind == "sand":
            cr.set_source_rgba(*hsv(p.hue, 0.45, 0.95), 0.85 * min(1, f * 4, (1 - f) * 8))
            cr.arc(p.x, p.y, p.size, 0, TAU)
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
        self.now_playing = NowPlaying(lookup_dates=not args.no_lookup)
        self.released = None
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
        np_ = self.now_playing
        self.blob.cool_target = 1.0 if np_.lanegan else 0.0
        self.blob.layne_target = 1.0 if np_.layne else 0.0
        self.blob.wright_target = 1.0 if np_.wright else 0.0
        self.blob.kurt_target = 1.0 if np_.kurt else 0.0
        self.blob.desert_target = 1.0 if np_.garcia else 0.0
        self.blob.tool_target = 1.0 if np_.tool else 0.0
        self.blob.korn_target = 1.0 if np_.korn else 0.0
        self.blob.nin_target = 1.0 if np_.nin else 0.0
        self.blob.linkin_target = 1.0 if np_.linkin else 0.0
        self.released = np_.released
        year = int(self.released[:4]) if self.released and self.released[:4].isdigit() else 0
        self.blob.era90_target = 1.0 if 1990 <= year <= 1999 else 0.0
        self.blob.y2k_target = 1.0 if 2000 <= year <= 2004 else 0.0
        self.blob.rage_target = 1.0 if np_.rage else 0.0
        self.blob.acid_target = 1.0 if np_.acid_bath else 0.0
        self.blob.nola_target = 1.0 if np_.nola else 0.0
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
        self.blob.draw_desert_sun(cr, self.get_allocated_width(), self.get_allocated_height(), self.audio)
        self.blob.draw_flash(cr)
        if self.show_scope:
            self.draw_scope(cr)
        self.blob.draw(cr, self.audio)
        self.draw_era_stamp(cr)
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

    def draw_era_stamp(self, cr):
        """90s songs get a VHS camcorder overlay (PLAY, tape counter, recording date); early-2000s
        songs the orange date a point-and-shoot camera burns into the photo. The date is the
        song's release date."""
        b = self.blob
        if (b.era90 < 0.01 and b.y2k < 0.01) or not self.released:
            return
        W, H = self.get_allocated_width(), self.get_allocated_height()
        parts = self.released.split("-")
        year = parts[0]
        month = int(parts[1]) if len(parts) > 1 else 0
        day = int(parts[2]) if len(parts) > 2 else 0
        fs = max(14, H * 0.032)
        pad = fs * 1.2
        cr.save()
        cr.select_font_face("Monospace", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        cr.set_font_size(fs)

        def text(x, y, s, rgb, a, right=False):
            if right:
                x -= cr.text_extents(s).x_advance
            cr.move_to(x + fs * 0.08, y + fs * 0.08)
            cr.set_source_rgba(0, 0, 0, 0.6 * a)        # drop shadow
            cr.show_text(s)
            cr.move_to(x, y)
            cr.set_source_rgba(*rgb, a)
            cr.show_text(s)

        k = b.era90
        if k > 0.01:
            # VHS: a little horizontal tracking jitter now and then
            jx = random.uniform(-3, 3) if random.random() < 0.08 else 0.0
            text(pad + jx, pad + fs, "PLAY", (1, 1, 1), 0.9 * k)
            x = pad + jx + cr.text_extents("PLAY ").x_advance
            cr.move_to(x, pad + fs * 0.18)
            cr.line_to(x + fs * 0.75, pad + fs * 0.58)
            cr.line_to(x, pad + fs * 0.98)
            cr.close_path()
            cr.set_source_rgba(1, 1, 1, 0.9 * k)
            cr.fill()
            el = int(time.monotonic() - self.now_playing.track_start)
            text(pad + jx, H - pad, f"SP  {el // 3600}:{el // 60 % 60:02d}:{el % 60:02d}",
                 (1, 1, 1), 0.9 * k)
            months = "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()
            date = " ".join(p for p in ((months[month - 1] + ".") if month else "",
                                        str(day) if day else "", year) if p)
            text(W - pad + jx, H - pad, date, (1, 1, 1), 0.9 * k, right=True)
        k = b.y2k
        if k > 0.01:
            stamp = f"'{year[2:]} {month if month else ''} {day if day else ''}".strip()
            stamp = " ".join(stamp.split())
            cr.set_font_size(fs * 1.1)
            x = W - pad * 1.5 - cr.text_extents(stamp).x_advance
            y = H - pad * 2.2
            cr.move_to(x, y)
            cr.text_path(stamp)
            cr.set_source_rgba(1.0, 0.4, 0.05, 0.3 * k)      # soft glow around the digits
            cr.set_line_width(fs * 0.18)
            cr.set_line_join(cairo.LINE_JOIN_ROUND)
            cr.stroke_preserve()
            cr.set_source_rgba(1.0, 0.58, 0.1, 0.95 * k)
            cr.fill()
        cr.restore()

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
    ap.add_argument("--no-lookup", action="store_true",
                    help="don't look up songs' release years on MusicBrainz "
                         "(the blob then can't tell 90s and early-2000s songs apart)")
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
