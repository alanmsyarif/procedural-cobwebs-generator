# Arachne

Procedural spider webs for Blender with a GPU-accelerated tearing physics
solver. Generate orb webs or chaotic corner cobwebs, simulate them with
real silk-like physics running on your GPU (no dependencies), and render
them as silk strands with dew droplets all from one panel.

Minimum Blender 5.2 LTS Version

## Features

**Two web generators**
- **Orb Web** classic radial/spiral webs with scalloped sag, uneven
  spiral spacing, damage gaps, asymmetry, wavy radials, and tangle
  threads for a natural aged look.
- **Chaotic Cobweb** spider-spun 3D cobwebs anchored to your scene
  geometry, based on the construction from Pixar's *Dust and Cobwebs for
  Toy Story 4* (SIGGRAPH 2019). Select a corner, prop, or room, and
  simulated spiders spin threads between the surfaces. **Clumping**
  draws spiders toward a few random attractor spots, knitting dense
  knots with sparser spans between the uneven density of real webs.
  **Bridge Bias** sets how much the anchors commit to spanning gaps:
  0 = local webbing hugging each surface, 1 = long cables strung
  between separate/floating pieces of geometry (a bridge).

- **Web Shot** Spider-Man style strands fired from an emitter over
  time. Each shot leaves the emitter at its own frame, flies at **Shot
  Speed** and sticks where it hits the selected geometry (a miss keeps a
  free end that whips). The emitter's animated location is sampled per
  shot, so a moving hand leaves each strand anchored where it fired;
  point an **Aim Target** at something to fire at it, or leave both
  empty to spray from the 3D cursor in all directions. An Aim Target is
  always part of the hit test, selected or not, and it sets the range:
  shots reach it however far away it is, and the ones the spread throws
  wide stop level with it instead of sailing past — so nothing runs
  through the thing being shot at. Silk that the lob or the clot would
  otherwise bury in a surface is lifted back out of it. Where a shot
  lands it splays into an **impact splat**: radial filaments sprayed
  across the surface, forked and knitted together, tips stuck down and
  wrapping over corners. **Cross Threads** string silk between shots
  already in the air, **Whip** bends the strands in flight. Unfired
  strands are frozen by the solver until their shot goes off, so nothing
  has sagged before it flies.

  The emitter fires more than once: **Bursts** volleys of **Shots**
  each, **Burst Gap** frames apart, every volley built from the
  emitter's location at the moment it goes off with its own clot,
  lashing and cross threads so a moving hand leaves a separate web per
  shot instead of one smeared fan. **Stick To Emitter** anchors the
  muzzle end to the emitter itself, the same way an impact point sticks
  to the geometry it hit: from the frame a strand fires, its near end is
  carried along as the emitter moves and turns, so the silk trails the
  hand that shot it (GPU solver; needs no collider).

**Live Update** toggle it on (Generate panel) after generating a web and
every parameter above rebuilds the web instantly as you drag the
sliders dial in the look in realtime, then add the solver and strands.
Moving the Emitter or the Aim Target counts as a change too — where the
emitter is decides where every strand starts, which way it flies and what
it hits — so dragging either one in the viewport rebuilds the Web Shot on
the spot. With Live Update off, the solver still keeps the muzzle anchors
on the emitter as you drag it.
Rebuilding a web that already has the solver on it restarts the
simulation from the new mesh at the current frame (otherwise the old
simulation would keep overwriting the display and the change would look
like it did nothing). Muzzle anchors are placed straight from the
emitter, so they land on it whatever frame you are on — but the rest of
the silk starts from its built pose, so play from frame 1 once the look
is dialled in.

**GPU physics solver** (built on Blender's native GPU module nothing to
install)
- Verlet / PBD solver with tearing: threads snap when overstretched
- Silk-like unilateral constraints (threads pull, never push)
- Tension control taut webs or drooping catenaries
- World-space gravity and wind with turbulence
- Collision: bounding sphere (fast) or full mesh via baked SDF, against
  a single object or an entire **collision collection** (all its meshes
  merged into one field)
- **Stickiness**: threads adhere to the collider on contact a
  tunable fraction of contacting points latch to the surface so the web
  drapes and clings instead of sliding off
- **Stuck Follows Collider**: anchors sitting on a collider a web
  shot's impact points, or contacts caught by Stickiness are carried
  along by its rigid motion, rotation included, so the web travels with
  the object instead of hanging where it stuck. Works per object in a
  collider collection: each anchor rides the member it is stuck to, so
  the web stretches as the pieces move apart. Collision itself is still
  a static bake for a collection only the attachments track motion
- Deteriorate (pre-broken threads) and pre-warm (starts settled)
- Real-time playback in the viewport
- Render-safe frame cache: every frame simulated in the viewport is
  cached and replayed during F12/animation renders (GPU compute can't
  run on the render thread) play through the range once to "bake"

**Rendering**
- Strandify: converts the simulated web to smooth silk tubes
  (Catmull-Rom smoothing, noisy radius)
- Dew droplets with physics: they cling to strands, condense and grow,
  slide down to hang under the silk, then drip off and free-fall once
  heavy enough respawning at their birth spot for a perpetual drip
  cycle. Droplets on torn strands are flung off. Use **Bake Dew for
  Render** (Render section of the panel) before rendering animations
  the render depsgraph can't reuse the viewport's live simulation, so
  the bake stores it to disk (and fills the web solver's render cache
  at the same time).
- Tension heatmap material visualize stretch from blue (rest) to red
  (about to tear)

## Installation

1. Download the latest `arachne.zip` from Releases.
2. Blender → Edit → Preferences → Add-ons → Install from Disk.
3. Enable **Arachne**. The panel appears in the 3D View sidebar
   (`N`) under **Arachne**.

## Quick start

1. (Optional) select a mesh to act as the collider or, in Chaotic
   Cobweb mode, the geometry the web anchors to.
2. Arachne panel → **Create Web + Sim + Strands**.
3. Rewind to frame 1 and press play. Push the collider through the web.

Anchor points are pinned automatically. To change them, enter Edit Mode,
select vertices, and use **Pin / Unpin** in the panel.

## Tips

- **Tension** ~0.95 for taut structural webs, ~0.5 for droopy abandoned
  ones. Pair low tension with **Deteriorate** for the aged look.
- **Spread** (Chaotic Cobweb) controls how uniformly spiders fill the
  volume: 0 = dense local clumps, 1 = Pixar-style even coverage. For
  dense reference-style webs try 1200 Spin Steps, Spread 0.9.
- **Web Shot**: set Shot Speed against your frame rate — 60 m/s at 24 fps
  crosses 2.5 m per frame, so short shots land in one or two frames. Pull
  it down to 10-20 to actually see the strand travel. Keep the solver's
  **Pre-warm Frames** at 0 (adding the solver to a shot web does this,
  along with Tension 0.95 — slack silk sags off the wall it just stuck
  to) so the first shots are not already settled at frame 1, and play
  from frame 1: the reveal is driven by the current frame, not the sim.
- A clot that unravels a few seconds after it fires is tearing, not the
  Clot settings: the cord is built to Clot Thickness for every burst, but
  once the strands' own segments snap the bundle spreads. The oldest
  burst goes first because it has hung the longest. Raise **Tear
  Threshold** (1.5 -> 4-6) or turn Enable Tearing off, and/or lower
  Tension so the silk carries slack instead of stretching to the limit.
- Web trailing a moving hand: keyframe the Emitter, leave **Stick To
  Emitter** on and play from frame 1. A muzzle holds its built pose until
  its own shot fires, then sits on the emitter from that frame on — it is
  placed from the emitter's location rather than simulated, so scrubbing,
  jumping and dropped frames all put it in the right place. The silk
  hanging off it still needs the frames played through.
- A single splat on a wall, as in the reference: Shots 1, Spread 0°,
  Shot Speed ~20, an Emitter and an Aim Target, Splat Size to taste. The
  surface being shot at should NOT be the solver's Collider — its
  bounding sphere swallows the web and tears it apart. Shot mode leaves
  the collider unset for that reason.
- **Mesh (SDF)** collision needs a closed mesh with outward normals.
  Animated collider *location* is supported; rotation is frozen at the
  bake. Raise SDF Resolution for thin or detailed colliders.
- Enable **Show Tension** on the Strandify modifier to see the stretch
  heatmap while tuning the Tear Threshold.
- If a fast collider tunnels through without tearing, raise **Substeps**.

## Limitations

- The GPU solver runs during viewport playback and UI renders. For
  command-line / farm rendering, play the shot through once and export
  the animated web (e.g. Alembic) first.
- Self-collision between threads is not simulated.
- SDF collision approximates rigid colliders; deforming meshes fall back
  to the bounded shape at bake time.

## Credits

- Cobweb construction and physics adapted from **Chang & Luoh, "Dust and
  Cobwebs for Toy Story 4"** (SIGGRAPH Talks 2019) and **Thomas Kole's
  Geometry Nodes Cobwebs**.
- Built by Amsy, with Claude (Anthropic).

## License

MIT do whatever, credit appreciated.
