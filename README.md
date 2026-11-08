# soundblob

A colorful, weird, googly-eyed blob that bounces around your desktop to whatever
music is playing. It's a transparent, click-through overlay for sway and other
wlroots compositors.

## Features

- Listens to system audio (the default sink's monitor, via `parec`).
- Detects kicks, snares and hi-hats separately, tracks tempo, and spots breakdowns and drops.
- Mood ranges from lazy (droopy eyelids, yawns) to crazy (spinning pupils, teeth, strobing).
  It follows loudness, drum density, tempo and guitar distortion.
- A rainbow oscilloscope line across the middle of the screen shows the live waveform.
- Genre flavours:
  - **Rock/metal:** headbangs on the snare, grows devil horns when guitars are distorted,
    rumbles on double bass. When it gets really heavy, the body turns dark, the brows go
    red and angry, an inverted pentagram glows on its forehead, flaring with the drums,
    and darkness creeps in from the screen edges.
  - **Techno/psytrance:** jumps that land on the kick, hovers during breakdowns, spiral eyes
    in build-ups, and explodes on the drop.
  - **Mellow (acoustic, ballads, quiet grunge):** calm music with clean guitars makes it
    smile with rosy cheeks, rock its head and drift side to side with the beat, bop instead
    of jumping, close its eyes in bliss now and then, and float little music notes. Distorted
    guitars or high energy break the spell.

## Requirements

Python 3, numpy, PyGObject (GTK 3), pycairo, gtk-layer-shell, and `parec`
(from libpulse or pipewire-pulse).

## Usage

```sh
./soundblob.py                  # run
./soundblob.py --size 100       # bigger blob (default radius 70 px)
./soundblob.py --layer bottom   # only on the bare desktop, behind windows
./soundblob.py --confetti 0.5   # less confetti (0 = none, 1 = default, 2 = double)
./soundblob.py --no-fps         # start with the FPS readout hidden
./soundblob.py --no-scope       # start with the oscilloscope hidden
./soundblob.py --no-vignette    # never darken the screen edges
./soundblob.py --debug          # print what the analyzer hears once a second
```

The FPS readout in the bottom-right corner shows frames per second, CPU draw time per
frame, and how much confetti is on screen. It turns amber below 50 fps and red below 30.
Confetti is the main CPU cost (about 0.02 ms per particle per frame), so lower `--confetti`
if the readout goes amber. The dark screen edges cost about 3 ms per frame at 1080p
while the music is heavy (nothing otherwise); `--no-vignette` turns them off.

### Sway keybindings

The overlay never takes keyboard focus, so hotkeys go through sway. Super+Shift+B toggles
the blob on and off. Super+Shift+F shows or hides the FPS readout (it sends `SIGUSR1`), and
Super+Shift+O shows or hides the oscilloscope (it sends `SIGUSR2`):

```
bindsym $mod+Shift+b exec pkill -f '^python3 \S*soundblob\.py( |$)' || /path/to/soundblob.py
bindsym $mod+Shift+f exec pkill -USR1 -f '^python3 \S*soundblob\.py( |$)'
bindsym $mod+Shift+o exec pkill -USR2 -f '^python3 \S*soundblob\.py( |$)'
```

The pattern is anchored on purpose. A loose `pkill -f soundblob.py` also matches the
`sh -c` that sway runs the binding in, plus any editor that has the file open.

Calibration constants (mood weights, distortion thresholds, kick/snare gating) are
at the top of `soundblob.py`.

## License

soundblob is free software, licensed under the
[GNU General Public License v3.0 or later](LICENSE).
