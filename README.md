# soundblob

A colorful, weird, googly-eyed blob that bounces around your desktop to whatever
music is playing. It's a transparent, click-through overlay for sway and other
wlroots compositors.

## Features

- Listens to system audio (the default sink's monitor, via `parec`).
- Detects kicks, snares and hi-hats separately, tracks tempo, and spots breakdowns and drops.
- Mood ranges from lazy (droopy eyelids, yawns) to crazy (spinning pupils, teeth, strobing).
  It follows loudness, drum density, tempo and guitar distortion.
- Genre flavours:
  - **Rock/metal:** headbangs on the snare, grows devil horns when guitars are distorted,
    rumbles on double bass.
  - **Techno/psytrance:** jumps that land on the kick, hovers during breakdowns, spiral eyes
    in build-ups, and explodes on the drop.

## Requirements

Python 3, numpy, PyGObject (GTK 3), pycairo, gtk-layer-shell, and `parec`
(from libpulse or pipewire-pulse).

## Usage

```sh
./soundblob.py              # run
./soundblob.py --size 100   # bigger blob
./soundblob.py --layer bottom   # only on the bare desktop, behind windows
./soundblob.py --debug      # print what the analyzer hears once a second
```

To add a sway toggle keybinding:

```
bindsym $mod+Shift+b exec pkill -f '^python3 \S*soundblob\.py( |$)' || /path/to/soundblob.py
```

Calibration constants (mood weights, distortion thresholds, kick/snare gating) are
at the top of `soundblob.py`.
