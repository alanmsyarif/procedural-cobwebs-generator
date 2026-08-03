# Changelog

## 4.3.1

Install fix only, no behaviour changes.

- **Installs as a Blender extension.** Arachne was a legacy script add-on, so
  Blender took the Python module name from the zip's top-level folder.
  GitHub's source zip is named after the repo and tag, e.g.
  `procedural-cobwebs-generator-4.3.0`, which is not a valid module name, and
  installing it failed with
  `No module named 'procedural-cobwebs-generator-4'`. `blender_manifest.toml`
  pins the module name to `arachne`, so any zip installs.
- **Uninstall moved.** Extensions put it in the `v` menu at the right-hand
  end of the add-on's row, not on an inline button.
- **Handler cleanup matched too broadly.** As an extension `__package__` is
  `bl_ext.<repo>.arachne`, and only the first component was compared, so
  disabling Arachne could strip other extensions' frame handlers.

## 4.0.0

First release since 3.2.3. Web Shot is now the default mode and got most of
the work: a second silk material, two-way rigid body coupling, multi-target
aiming, and a run of tearing fixes.

Major, not minor: a public operator and two properties were removed, and
existing scenes need their Strandify modifier re-added. See **Upgrading**.

### Breaking

- **`arachne.full_setup` removed.** The "Create Web + Sim + Strands" button
  is gone; use Generate Web → Add GPU Solver → Add Strandify. Any keymap or
  script calling that operator will error.
- **`collider_mass` and `collider_grip` removed.** The web-pull feature now
  drives Blender's rigid body sim, so mass and friction come from the
  collider's Physics tab. Values saved in a .blend are silently dropped.
- **Strandify node group is at version 12.** Existing modifiers keep the old
  group, renamed `.old`. Delete and re-add the modifier to get dew shade
  smoothing and the Synth Web socket.
- **The Synth-Web material rebuilds itself in place** when its stored version
  is behind. Hand edits to that node tree are lost on upgrade. Done in place
  rather than aside so every object already pointing at it picks up the new
  look.

### Added

- **Synth Web material** synthetic web-fluid, shaded to the published
  chemistry: shear-thinning, virtually solid at rest and turned fluid only by
  being fired, knitting into a nylon-related fibre on contact with air.
  Adhesion fades as it sets, driven per point by how long a thread has been in
  the air (`swf_shot_t` against the current frame): fresh thread reads wet and
  tacky, then goes matte and opaque, then blooms with ester powder. Timed by a
  **Cure Frames** value node, 36 by default. Toggle **Synth Web** on the
  Strandify modifier.
- **Web Pulls Collider** two-way coupling. Threads stuck to the collider
  haul it around instead of only being held by it. The web's net pull on its
  stuck points is summed and fed into Blender's rigid body sim as a wind
  field, so mass, friction, gravity and tumble are Bullet's. Run **Set Up
  Rigid Body Pull** once; **Static** makes the body Passive; **Pull Strength**
  scales stretch into force.
- **Aim Collection** fire at several objects, one per burst in turn.
  Overrides the single Aim Target. Each volley reaches only its own target, so
  clustered targets give separate webs rather than cords strung between
  neighbours. Members no burst reaches are ignored entirely.
- **Advanced section** in the Web Shot panel, holding Arc, Whip, Slack and the
  three Splat controls.
- **Real solver error reporting.** A caught exception used to disable the GPU
  backend for the session and say only "GPU compute unavailable". The panel now
  names the actual exception, prints a traceback to the console, distinguishes
  a build without compute shaders from a session-only failure, and offers a
  **Clear Error** button.

### Changed

- **Tension goes to 2.0.** Above 1.0 it pre-tensions the silk — rest lengths
  shorter than built so the last of the gravity droop comes out. Tearing
  measures strain against rest, so this costs tear margin; the panel prints
  what is left.
- **Cross Threads cap 200 → 4000** (slider stops at 400, type for more). The
  nearest-neighbour search switched from a full sort to a partition, since only
  the three nearest are used.
- **Detail sizes the opened fan only** on a Web Shot. The travelling clot takes
  its point density from Clot Twist instead, at about eight points per turn
  it is a helix, and at the old shared density it read as a zigzag. Point
  counts are per unit of flight, so a fan segment is the same length whatever
  the Clot fraction.
- **The clot meanders instead of bowing.** Three wave components over a wider
  band at double the amplitude, so the cord kinks two or three times along its
  length rather than making one broad bend. This is what it does at Whip 0.
- **Live Update stands down while the solver is showing**, rather than
  restarting the simulation on every mouse move. Switch the Arachne GPU Apply
  modifier's viewport display off to tweak live with the solver's settings
  intact; switch it back on and the sim resumes next frame.
- **Dew is off by default** it is a simulated point cloud on every strand, so
  it is opt-in now rather than something to turn off.
- **Dew droplets are shade smooth.** A subdivision-2 icosphere read as a
  faceted rock at droplet scale.
- **Live Update is on by default.**
- **Range greys out when an aim target is set** the target sets the reach
  itself, so Range has nothing to say.
- Web Shot defaults reworked: mode is Web Shot, First Shot Frame 15, Shots 36,
  Shot Interval 0, Spread 0°, Clot Thickness 0.005, Clot Twist 4.0, Arc 0,
  Whip 0, Splat off, Cross Threads 200.
- Panel warns when a bounding-sphere collider will reach past the flat faces
  the web is stuck to, and when an Aim Collection holds more targets than
  Bursts.

### Fixed

- **Dew droplets jittered while sliding.** Two causes: the tangential gravity
  vector was never normalised, so slide speed swung with whichever tube facet
  a droplet sat on; and droplets were snapped exactly onto a six-sided tube,
  landing on facet boundaries where the nearest-face query flip-flopped frame
  to frame. Now normalised, faded out near the poles where the tangent is
  numerically meaningless, and relaxed toward the surface instead of snapped.
- **Web shots tore themselves apart.** The clot's own strand segments are
  short by design braid resolution and tearing measures strain against the
  segment, so any jostle snapped the rope along its length. Those segments are
  now flagged unbreakable at build time, the same treatment the binder threads
  already had.
- **Only the last burst's clot was protected from tearing.** The unbreakable
  edge list was rebuilt per volley, so with Bursts > 1 every earlier clot burst
  apart on its own.
- **Rigid body pull applied before the web arrived.** A Web Shot's impact
  anchors are pinned to the collider from frame 1, long before the flying tip
  reaches them, so the whole web hauled on the object while still visibly in
  the air. Gated on the same reveal rule the geometry uses. The pull field also
  kept its last strength through a reset or a scrub, so a replay started with
  the object already moving under a force nothing was exerting.
- **Cross threads could link a strand to itself.** The nearest-point pick took
  three indices unconditionally, so when nearly every candidate belonged to the
  source's own strand it could return one at infinite distance.
- **Live Update restarted the simulation on every collider drag.** Dragging an
  aim target or collider is a depsgraph tick per mouse move; each one rebuilt
  the mesh and dropped the solver state.
- Unused Aim Collection members no longer act as obstacles, and targets no
  longer catch shots aimed at each other.

### Upgrading

1. Reload scripts, or restart Blender. The panel should read
   `Build 4.0.0 multi-target`; an older number there means Blender is still
   running a cached copy.
2. On existing webs, delete and re-add the **Arachne Strandify** modifier
   the node group is at version 12 and old modifiers keep the previous one.
3. Regenerate any Web Shot. Strand point layout and the unbreakable-edge flags
   are baked at build time, so **Reset GPU Sim** is not enough.
4. If you used web pull with the old `collider_mass` / `collider_grip`, set
   mass and friction on the collider's Physics tab instead and run **Set Up
   Rigid Body Pull**.
