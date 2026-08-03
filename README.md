# Arachne

Procedural spider webs for Blender, with a GPU tearing solver. Generate a
web, simulate it with silk-like physics on your GPU, render it as strands
all from one panel. No dependencies.

Minimum Blender 5.2 LTS.

## Install

1. Download `arachne.zip` from Releases.
2. Edit → Preferences → Add-ons → Install from Disk.
3. Enable **Arachne**. Panel appears in the 3D View sidebar (`N`).

> Use the `arachne.zip` release asset, not GitHub's green **Code → Download
> ZIP** button. That one wraps the files in a folder named after the repo and
> tag, e.g. `procedural-cobwebs-generator-4.3.0`, and Blender fails with
> `No module named 'procedural-cobwebs-generator-4'` — hyphens and dots are
> illegal in a Python module name.

## Building the zip

Arachne is a script add-on, so the zip must hold exactly one top-level
folder named `arachne` — that folder name becomes the Python module name
Blender imports.

```sh
git archive --prefix=arachne/ -o arachne.zip HEAD \
  ':!images' ':!test_shot.py' ':!.claude'
```

Run it in the repo root. Archives the current commit, so commit before
building. `.git` is left out automatically.

## Quick start

1. *(Optional)* Select a mesh, the collider, or in Chaotic Cobweb mode the
   geometry the web anchors to.
2. **Generate Web**, then **Add GPU Solver**, then **Add Strandify**.
3. Rewind to frame 1, press play. Push the collider through the web.

Anchors are pinned automatically. To change them: Edit Mode → select
vertices → **Pin / Unpin**.

## Web types

**Orb Web** — classic radial/spiral web. Scalloped sag, uneven spiral
spacing, damage gaps, asymmetry, wavy radials, tangle threads.

**Chaotic Cobweb** — spider-spun 3D webbing anchored to your scene
geometry (Pixar *Toy Story 4* construction). Select a corner, prop or room
and simulated spiders spin between the surfaces.

- **Spread** 0 = dense local clumps, 1 = even coverage.
- **Clumping** draws spiders to random attractors: dense knots, sparse
  spans between.
- **Bridge Bias** 0 = webbing hugging each surface, 1 = long cables
  strung between separate pieces.

**Web Shot** strands fired from an emitter over time. Each shot leaves at
its own frame, flies at **Shot Speed**, sticks where it hits. A miss keeps
a free end that whips.

- **Emitter** animate it and each volley is read at the frame it goes off,
  location and aim direction both, so a moving hand leaves every strand
  anchored where it fired and pointing where it pointed. Unset = 3D cursor.
- **Aim Target** fire at it. Always part of the hit test, and it sets the
  range: shots reach it however far it is, and wide ones stop level with it
  instead of sailing past. Animated targets are aimed at — and hit — where
  they will be at that volley's frame, not where the playhead left them.
- **Aim Collection** overrides Aim Target and fires at each member in turn,
  one per burst: two objects across three bursts goes first, second, first.
  One target per volley keeps each burst reading as a single cord thrown at
  a single thing. **Raise Bursts to match the target count** members no
  burst reaches are ignored entirely, hit test included, so extras beyond
  Bursts do nothing. Each volley can only stick to its own target plus any
  mesh you had selected targets never block each other, so clustering them
  gives separate webs rather than cords strung between neighbours.
- **Bursts / Burst Gap** the emitter fires more than once. Each volley is
  built fresh from the emitter's location at that moment, with its own
  clot, lashing and cross threads.
- **Stick To Emitter** the muzzle end rides the emitter as it moves and
  turns, so the silk trails the hand that shot it. GPU solver, no collider
  needed.
- **Splat / Cross Threads / Whip** impact spray across the surface, silk
  strung between shots already in the air, in-flight bend.

Unfired strands are frozen until their shot goes off, so nothing has sagged
before it flies.

**Animated emitters and targets.** Reading location F-curves only sees a
plain keyframe, so the build steps the scene to each volley's own frame and
reads the transforms — and the target's collision geometry off the
evaluated depsgraph instead. Parenting to a rig or a bone, constraints, NLA
strips and drivers all work. A setup with nothing animated never sets a
frame and costs exactly what it did before.

## Live Update

On by default. Every parameter rebuilds the tracked web as you drag, so you
can dial in the look before simulating. Moving the geometry a web was built
from counts as a change too: the Emitter or Aim Target for a Web Shot (where
the emitter is decides where every strand starts), and the selected anchor
meshes for a Chaotic Cobweb move or scale a wall and the web re-spins onto
it, staying stuck to the corner instead of hanging in space.

Only a drag counts. Playing back or scrubbing moves an animated emitter on
every frame, and rebuilding there would be wasted work the build reads the
animation at the frames the volleys fire on, so the web is the same one
whatever frame you are parked on.

**It stands down while the solver is showing.** Regenerating swaps the mesh
out and would restart the sim from scratch on every mouse move, so the panel
shows a lock instead.

To get back to tweaking, switch the **Arachne GPU Apply** modifier's viewport
display off. That modifier is what puts simulated positions on screen with
it hidden you are looking at the built mesh, so rebuilds are safe and visible
again, and every solver setting stays put. Switch it back on and the sim
resumes on the next frame. No need to remove the solver.

With the solver running, dragging the collider or emitter still carries the
anchors along the web reacts without restarting.

## GPU solver

Built on Blender's native GPU module, nothing to install.

- Verlet / PBD with tearing: threads snap when overstretched
- Unilateral silk constraints (threads pull, never push)
- **Tension** 0 = drooping catenaries, 1 = taut (rest lengths as built),
  above 1 pre-tensions the silk so the last of the gravity droop comes out
  (2 = rest lengths 15% shorter than built)
- World-space gravity and wind with turbulence
- Collision: bounding sphere (fast) or baked mesh SDF, against one object
  or a whole **collider collection** (merged into one field)
- **Stickiness** a tunable fraction of contacting points latch to the
  surface, so the web drapes and clings instead of sliding off
- **Stuck Follows Collider** anchors on a collider ride its rigid motion,
  rotation included. Per object in a collection, so the web stretches as
  the pieces move apart. Collision for a collection is still a static bake;
  only the attachments track motion.
- **Web Pulls Collider** two-way coupling into Blender's rigid body sim.
  Threads stuck to the collider haul it around instead of only being held
  by it: fire a shot at a prop and drag it over. Mass, friction, gravity
  and tumble are Bullet's, straight off the Physics tab; **Pull Strength**
  scales thread stretch into force, and **Static** makes the body Passive.
  Hit **Set Up Rigid Body Pull** once, then play from frame 1 see
  [Tips](#tips).
- **Deteriorate** (pre-broken threads), **Pre-warm** (starts settled)
- Render-safe frame cache. see [Rendering](#rendering)

## Rendering

**Strandify** converts the simulated web to smooth silk tubes (Catmull-Rom,
noisy radius). Three materials, switched on the modifier:

| | |
|---|---|
| **Silk** (default) | Dusty translucent natural silk, visible at grazing angles |
| **Synth Web** | Synthetic web-fluid: shear-thinning, virtually solid at rest and turned fluid only by being fired, knitting into a nylon-related fibre on contact with air. Adhesion fades as it sets fresh thread reads wet and tacky, then goes matte and opaque, then blooms with ester powder. Timed by **Cure Frames** in the node tree (36 = 1.5 s at 24fps) |
| **Show Tension** | Stretch heatmap, blue (rest) to red (about to tear) |

**Dew droplets** are off by default tick **Enable Dew**. They cling to
strands, condense and grow, slide under the silk, then drip off and
free-fall once heavy enough, respawning at their birth spot. Droplets on
torn strands are flung off.

**Before rendering animations**, play the frame range through once in the
viewport, then hit **Bake Dew for Render**. GPU compute can't run on the
render thread, so the solver replays a viewport cache; the dew sim needs
its own disk bake. The bake fills both.

## Tips

- **Tension** ~0.95 for taut structural webs, ~0.5 for droopy abandoned
  ones. Pair low tension with **Deteriorate** for the aged look. Rest
  lengths bake at sim start reset or return to frame 1 after changing it.
- Going above Tension 1 shortens rest lengths, so threads start closer to
  snapping. Raise **Tear Threshold** to match.
- **Shot Speed** against your frame rate: 60 m/s at 24 fps crosses 2.5 m
  per frame, so short shots land in one or two frames. Drop to 10–20 to
  actually see the strand travel.
- Play Web Shots **from frame 1** the reveal is driven by the current
  frame, not the sim. Keep **Pre-warm Frames** at 0 (adding the solver to a
  shot web does this, along with Tension 0.95) so early shots aren't
  already settled.
- A clot that unravels seconds after firing is *tearing*, not the Clot
  settings the oldest burst goes first because it has hung longest. Raise
  **Tear Threshold** (1.5 → 4–6), or turn tearing off, or lower Tension.
- Single splat on a wall: Shots 1, Spread 0°, Shot Speed ~20, an Emitter
  and an Aim Target, Splat Size to taste. The wall must **not** be the
  solver's Collider its bounding sphere swallows the web and tears it
  apart. Shot mode leaves the collider unset for this reason.
- **Mesh (SDF)** collision needs a closed mesh with outward normals.
  Animated *location* is supported; rotation freezes at the bake. Raise SDF
  Resolution for thin or detailed colliders.
- Fast collider tunnelling through without tearing → raise **Substeps**.
- **Web Pulls Collider** works through anchors welded to the collider, so
  it needs **Stuck Follows Collider** on without that nothing is attached
  to pull with. Run **Set Up Rigid Body Pull** once (it makes the collider
  an Active rigid body and builds the force field), then **play forward
  from frame 1**: Bullet won't re-simulate frames its cache already holds,
  and the pull lands a frame after it is computed. Nothing moving? Raise
  **Pull Strength**, or lower Mass / Friction on the Physics tab.
- The pull is delivered as a wind field scoped to the collider's bounding
  radius, because Blender exposes no way to push a single rigid body from
  Python. Another Active rigid body sitting inside that radius will feel it
  too. Deleting the "Arachne Pull" empty switches the coupling off.
- Panel showing an older build than the Add-ons list means Blender is
  running a cached copy. Reload scripts, or restart.

## Limitations

- The solver runs during viewport playback and UI renders. For
  command-line / farm rendering, play through once and export the animated
  web (e.g. Alembic) first.
- No self-collision between threads.
- SDF collision approximates rigid colliders; deforming meshes fall back to
  the bounded shape at bake time.
- Disable Persistent data in render, so the simulation doesn't stop

## Credits

Cobweb construction and physics adapted from Chang & Luoh, **"Dust and
Cobwebs for Toy Story 4"** (SIGGRAPH Talks 2019) and **Thomas Kole's
Geometry Nodes Cobwebs**.

Built by Amsy, with Claude (Anthropic).

## License

MIT — do whatever, credit appreciated.
