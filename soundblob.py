#!/usr/bin/env python3
"""
soundblob — a colorful, weird, googly-eyed blob that bounces around your
desktop to whatever music is playing.

* Captures system audio from the default sink monitor (PulseAudio/PipeWire via parec)
* FFT + beat detection -> jumps, squish, spectrum-shaped wobble, particle bursts
* Transparent, click-through Wayland layer-shell overlay (sway/wlroots)

Quit with Ctrl+C in the terminal, or: pkill -f soundblob.py
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
from gi.repository import Gtk, Gdk, GLib  # noqa: E402

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
TRAIL_MAX = 18            # ghost-trail circles at full craziness (each is a big translucent fill)

# Mood calibration, measured on real rock/metal recordings and synthetic techno/psytrance.
# Spectral flatness (200 Hz-5 kHz): clean/acoustic guitars 0.27-0.52, distorted metal 0.67-0.73
FLAT_CLEAN, FLAT_DIRTY = 0.50, 0.68
W_LOUD = 0.30      # loudness vs. the session's loud level (quiet intros/ballads -> lazy)
W_AGGR = 0.30      # distortion (rock/metal guitars)
W_TEMPO = 0.15     # fast, *reliable* tempo
W_DENSITY = 0.20   # drum hits per second
W_DRIVE = 0.20     # rock-steady fast grid: psytrance / hard techno
W_METAL = 0.20     # distortion + fast kicks: thrash / double bass / blast beats
KICK_LOWNESS, SNARE_LOWNESS = 8.0, 6.0


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
        self.ref_rms = 0.0         # "how loud is loud" for this listening session
        self.loud_rel = 1.0
        self.energy = 0.4          # 0 = lazy, 0.5 = normal, 1 = crazy
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
        self.buf = np.roll(self.buf, -len(samples))
        self.buf[-len(samples):] = samples
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
        """How hyped is the music right now? Loudness relative to the last
        ~30 s, drum density, tempo and distortion."""
        self.short_rms += (rms - self.short_rms) * 0.06          # ~0.2 s
        if self.short_rms > 0.003:
            if self.ref_rms == 0.0:
                self.ref_rms = self.short_rms
            # rises within ~10 s, forgets over ~3 min: a quiet intro after a loud
            # song (or a ballad on a metal album) reads as quiet
            k = 1 / (10 * FPS) if self.short_rms > self.ref_rms else 1 / (180 * FPS)
            self.ref_rms += (self.short_rms - self.ref_rms) * k
        self.loud_rel = self.short_rms / max(self.ref_rms, 1e-4)

        loud = clip01((self.loud_rel - 0.35) / 0.75)
        aggr_target = (clip01((self.flatness - FLAT_CLEAN) / (FLAT_DIRTY - FLAT_CLEAN)) *
                       clip01(self.high) * clip01(self.loud_rel * 1.5))
        self.aggression += (aggr_target - self.aggression) * 0.02
        steady = clip01((self.tempo_conf - 0.3) / 0.3) if self.bpm else 0.0
        tempo = clip01((self.bpm - 100) / 45) * steady
        drive = clip01((self.bpm - 126) / 16) * steady
        density = clip01((self.kick_rate + self.snare_rate + 0.5 * self.hat.rate +
                          0.3 * self.onset_rate - 2.5) / 5)
        metal = self.aggression * clip01((self.kick_rate - 1.5) / 2)
        raw = clip01(W_LOUD * loud + W_AGGR * self.aggression + W_TEMPO * tempo +
                     W_DENSITY * density + W_DRIVE * drive + W_METAL * metal)
        if self.breakdown:
            raw *= 0.75 + 0.5 * self.build      # tension rises with the build-up
        if self.short_rms < 0.003:
            raw = 0.0
        self.energy += (raw - self.energy) * (0.03 if raw > self.energy else 0.01)

    def debug_line(self):
        return (f"bpm={self.bpm:5.1f} conf={self.tempo_conf:.2f} kick/s={self.kick_rate:4.1f} "
                f"snare/s={self.snare_rate:4.1f} onset/s={self.onset_rate:4.1f} "
                f"flat={self.flatness:.3f} aggr={self.aggression:.2f} loud={self.loud_rel:.2f} "
                f"bd={'Y' if self.groove_break else ('i' if self.breakdown else '-')} build={self.build:.2f} "
                f"drops={self.drops} energy={self.energy:.2f}")


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
        self.rumble = 0.0          # double-bass / blast-beat vibration
        self.hover = 0.0           # floats during breakdowns (techno/psy)
        self.spiral = 0.0          # hypnotic eyes during build-ups (psy)
        self.charge = 0.0          # build-up tension
        self.drop_boost = 0.0      # forced craziness after a drop
        self.flash = 0.0           # screen flash on the drop
        self.land_at = 0.0         # when a beat-synced jump will touch down

    @property
    def lazy(self):
        return clamp01((0.38 - self.mood) / 0.3)

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
        self.nod = max(0.0, self.nod - dt * 7)
        self.flash = max(0.0, self.flash - dt * 3)

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
        elif L > 0.6 and not quiet and random.random() < dt * 0.07:
            self.yawn = 1.0

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

        for p in self.particles:
            p.life -= dt
            p.vy += (500 if p.kind != "ring" else 0) * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.rot += dt * 6
        self.particles = [p for p in self.particles if p.life > 0]

    def emit(self, p):
        """Spawn a confetti particle, thinned by the confetti budget."""
        if len(self.particles) < self.max_particles and random.random() < self.confetti_rate:
            self.particles.append(p)

    def radius(self, bass):
        # inflates while a build-up charges
        return self.R * (1 + min(bass, 2.0) * 0.18) * (1 + 0.3 * self.charge)

    def on_kick(self, strength, W, H, a):
        if self.hover > 0.5:
            # floating in a breakdown: kicks just make it pulse
            self.squash_v += 1.5
            return
        self.jump(strength, W, H, a, synced=True)

    def on_snare(self, strength):
        """Headbang! Body nods down, face dips, little sideways shove."""
        s = min(strength, 3.0)
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
        # too lazy to jump for weak beats: just a squishy little bop in place
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
            cr.set_source_rgba(*hsv(th + f * 0.5 + 0.3, sat, 1), f * 0.22)
            cr.arc(tx, ty, tr * (0.35 + f * 0.6), 0, TAU)
            cr.fill()

        # particles behind
        for p in self.particles:
            self.draw_particle(cr, p)

        cr.save()
        cr.translate(*self.shake)
        pts = self.outline(a, r)

        # glow (bigger and brighter while a build-up charges)
        for k, al in ((1.45 + 0.4 * self.charge, 0.06 + 0.12 * self.charge), (1.22, 0.10)):
            cr.save()
            cr.translate(self.x, self.y)
            cr.scale(k, k)
            cr.translate(-self.x, -self.y)
            smooth_closed_path(cr, pts)
            cr.set_source_rgba(*hsv(self.hue + 0.5, sat), al + a.level * 0.08 + self.crazy * 0.08)
            cr.fill()
            cr.restore()

        # body
        smooth_closed_path(cr, pts)
        grad = cairo.RadialGradient(self.x - r * 0.35, self.y - r * 0.4, r * 0.05,
                                    self.x, self.y, r * 1.35)
        grad.add_color_stop_rgba(0, *hsv(self.hue + 0.12, 0.35 * sat, 1), 1)
        grad.add_color_stop_rgba(0.45, *hsv(self.hue, sat, 1), 1)
        grad.add_color_stop_rgba(1, *hsv(self.hue - 0.18, sat, 0.55), 1)
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
            cr.set_source_rgba(*hsv(self.hue + 0.25 + j * 0.13, 0.9 * sat, 1), 0.55)
            cr.arc(self.x + math.cos(ang) * dist, self.y + math.sin(ang) * dist, sr, 0, TAU)
            cr.fill()
        cr.restore()

        cr.set_line_width(4 + a.high * 4)
        cr.set_source_rgba(*hsv(self.hue + 0.5, 1, 0.25), 0.9)
        cr.stroke()

        # shine
        cr.set_source_rgba(1, 1, 1, 0.35)
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
        sq = self.squash
        face_y = (self.y - r * 0.12 * (1 - sq) + r * 0.08 * L   # lazy face slides down
                  + self.nod * r * 0.14)                       # headbang dip
        sleepy = self.sleepiness
        yawn_open = math.sin(math.pi * (1 - self.yawn)) if self.yawn > 0 else 0.0

        eyes = ((-0.36, -0.08, self.eye_sizes[0]), (0.34, -0.02, self.eye_sizes[1]))
        # droopy half-closed lids when lazy, wide open when crazy; squint on the headbang
        lid = max(self.blink, sleepy * 0.9, L * 0.5, yawn_open * 0.85, self.nod * 0.45) * (1 - self.spiral)
        for i, ((ex, ey, es), p) in enumerate(zip(eyes, self.pupils)):
            grow = 1 + 0.3 * C * (1 + 0.3 * math.sin(self.t * 20 + i * 2))
            ex_ = self.x + ex * r
            ey_ = face_y + ey * r
            er = r * 0.27 * es * grow
            self.draw_eye(cr, ex_, ey_, er, p, lid, a, idx=i)
            self.draw_brow(cr, ex_, ey_, er, r, side=-1 if i == 0 else 1, idx=i)

        if self.third_eye > 0.01:
            s = min(1, self.third_eye * 2)
            self.draw_eye(cr, self.x, face_y - r * 0.5, r * 0.16 * s, self.pupils[0], 0, a,
                          hue_eye=True, idx=2)

        # mouth: opens with volume; small & lopsided when lazy, huge grin when crazy
        mw = r * 0.42 * lerp(0.6, 1.25, m) * (1 - yawn_open * 0.45)
        mh = r * (0.04 + min(a.level, 1.2) * 0.32) * lerp(0.35, 1.5, m) * (1 - sleepy)
        mh = max(mh, r * 0.55 * yawn_open)
        my = face_y + r * 0.42
        tilt = r * 0.12 * L                      # one corner droops: "meh"
        steps = 16
        cr.save()
        cr.move_to(self.x - mw, my - tilt * 0.5)
        for i in range(steps + 1):
            u = i / steps
            x = self.x - mw + 2 * mw * u
            y = (my + (u - 0.5) * tilt + math.sin(u * math.pi) * mh +
                 math.sin(u * 18 + self.t * 20) * a.high * r * 0.03 * (0.3 + m))
            cr.line_to(x, y)
        for i in range(steps, -1, -1):
            u = i / steps
            x = self.x - mw + 2 * mw * u
            cr.line_to(x, my + (u - 0.5) * tilt - math.sin(u * math.pi) * mh * (0.25 + yawn_open * 0.6))
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

        if sleepy > 0.8:
            cr.select_font_face("Sans", cairo.FONT_SLANT_ITALIC, cairo.FONT_WEIGHT_BOLD)
            for i in range(3):
                ph = (self.t * 0.5 + i / 3) % 1
                cr.set_font_size(r * (0.2 + ph * 0.25))
                cr.set_source_rgba(*hsv(self.hue + i * 0.2), 1 - ph)
                cr.move_to(self.x + r * (0.6 + ph * 0.6), self.y - r * (0.7 + ph * 1.1))
                cr.show_text("z")

    def draw_brow(self, cr, x, y, er, r, side, idx):
        """Eyebrows: low and drooping when lazy, high and twitching when crazy."""
        L, C, m = self.lazy, self.crazy, self.mood
        lift = er * (lerp(-0.1, 0.45, m) + 0.25 * C * math.sin(self.t * 25 + idx * 1.7))
        by = y - er * 1.3 - lift
        w = er * 0.9
        outer_dy = er * (0.45 * L - 0.5 * C * (1 if idx == 0 else -0.4))
        ox, ix = x + side * w, x - side * w
        oy, iy = by + outer_dy, by - outer_dy * 0.3
        cr.move_to(ox, oy)
        cr.curve_to(ox - side * w * 0.5, by - er * 0.2, ix + side * w * 0.5, by - er * 0.2, ix, iy)
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        cr.set_line_width(max(3, r * 0.07))
        cr.set_source_rgba(*hsv(self.hue + 0.5, 1, 0.2), 0.95)
        cr.stroke()

    def draw_eye(self, cr, x, y, er, p, lid, a, hue_eye=False, idx=0):
        C, L = self.crazy, self.lazy
        cr.arc(x, y, er, 0, TAU)
        cr.set_source_rgba(1, 1, 1, 1)
        cr.fill_preserve()
        cr.set_line_width(max(2, er * 0.12))
        cr.set_source_rgba(0.1, 0, 0.1, 1)
        cr.stroke()
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
            if hue_eye:
                cr.set_source_rgba(*hsv(self.hue + 0.5), 1 - sp)
            elif self.horns > 0.3:
                # metal: glowing red pupils
                cr.set_source_rgba(0.55 * self.horns, 0.0, 0.08 * (1 - self.horns), 1 - sp)
            else:
                cr.set_source_rgba(0.05, 0.0, 0.08, 1 - sp)
            cr.arc(px, py, pr, 0, TAU)
            cr.fill()
            cr.set_source_rgba(1, 1, 1, 0.9 * (1 - sp))
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
            cr.set_source_rgba(*hsv(self.hue, 1, 0.8), 1)
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
        self.last_time = None
        # FPS counter: frames actually drawn, averaged over half-second windows
        self.show_fps = not args.no_fps
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
        self.queue_draw()
        return GLib.SOURCE_CONTINUE

    def on_draw(self, widget, cr):
        t0 = time.perf_counter()
        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.set_source_rgba(0, 0, 0, 0)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)
        self.blob.draw_flash(cr)
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
    Gtk.main()
    audio.running = False


if __name__ == "__main__":
    main()
