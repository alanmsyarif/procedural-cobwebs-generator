# Graph Report - .  (2026-07-25)

## Corpus Check
- 14 files · ~55,505 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 218 nodes · 383 edges · 11 communities
- Extraction: 87% EXTRACTED · 12% INFERRED · 1% AMBIGUOUS · INFERRED: 46 edges (avg confidence: 0.7)
- Token cost: 107,614 input · 0 output

## Community Hubs (Navigation)
- GPU Solver Operators
- Materials and Node Utilities
- Arachne Feature Concepts
- Web Mesh Generator
- Geometry Node Tree Helpers
- Add-on UI and Registration
- Native GPU Compute Backend
- Strandify and Dew Baking
- Viewport Render Screenshot
- Tearing Solver and Pinning
- Photoreal Cobweb Reference

## God Nodes (most connected - your core abstractions)
1. `H` - 32 edges
2. `NativeState` - 14 edges
3. `GPU Physics Solver` - 11 edges
4. `apply_strandify()` - 10 edges
5. `Chaotic Cobweb Generator` - 9 edges
6. `build_web_object()` - 7 edges
7. `gpu_backend_available()` - 7 edges
8. `_on_frame()` - 7 edges
9. `enable_gpu_solver()` - 7 edges
10. `ARN_OT_add_gpu_solver` - 7 edges

## Surprising Connections (you probably didn't know these)
- `ARN_GPUProps` --uses--> `H`  [INFERRED]
  gpu_solver.py → nodeutils.py
- `ARN_OT_add_gpu_solver` --uses--> `H`  [INFERRED]
  gpu_solver.py → nodeutils.py
- `ARN_OT_remove_gpu_solver` --uses--> `H`  [INFERRED]
  gpu_solver.py → nodeutils.py
- `ARN_OT_reset_gpu` --uses--> `H`  [INFERRED]
  gpu_solver.py → nodeutils.py
- `ARN_OT_pin_vertices` --uses--> `H`  [INFERRED]
  gpu_solver.py → nodeutils.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Render-time Bake Pipeline (GPU sim cannot run on the render thread)** — readme_gpu_physics_solver, readme_render_frame_cache, readme_bake_dew_for_render, readme_dew_droplets, readme_farm_render_limitation [INFERRED 0.85]
- **Aged / Natural Web Look Controls** — readme_deteriorate, readme_tension_control, readme_orb_web, readme_clumping, readme_spread [INFERRED 0.75]
- **Generate to Simulate to Render Flow** — readme_create_web_sim_strands, readme_chaotic_cobweb, readme_orb_web, readme_gpu_physics_solver, readme_strandify, readme_pin_unpin [EXTRACTED 1.00]
- **Anchor to Sim to Strand Cobweb Pipeline** — images_ss1_suzanne_target_mesh, images_ss1_anchor_point_attachment, images_ss1_radial_web_sector, images_ss1_cloth_sim_relaxation, images_ss1_strandify_curve_conversion, images_ss1_cobweb_render_showcase [INFERRED 0.75]
- **Visual Realism Cues of a Natural Cobweb** — images_ss2_irregular_web_topology, images_ss2_dew_droplet_beading, images_ss2_strand_catenary_sag, images_ss2_anchor_points_on_geometry [INFERRED 0.85]

## Communities (11 total, 0 thin omitted)

### Community 0 - "GPU Solver Operators"
Cohesion: 0.08
Nodes (27): _mark_broken(), NativeState, Simulation state: step() advances physics, write_back() -> mesh., ARN_GPUProps, ARN_OT_add_gpu_solver, ARN_OT_pin_vertices, ARN_OT_remove_gpu_solver, ARN_OT_reset_gpu (+19 more)

### Community 1 - "Materials and Node Utilities"
Cohesion: 0.11
Nodes (24): # NOTE: the `swf_` prefix on attributes and custom properties below is, ensure_dew_material(), ensure_silk_material(), ensure_tension_material(), Emissive heatmap driven by the solver's swf_tension attribute:     deep blue at, input_identifier(), minmax_sockets(), Interface identifier of a group input socket (for modifier IDProps). (+16 more)

### Community 2 - "Arachne Feature Concepts"
Cohesion: 0.12
Nodes (29): Arachne, Bake Dew for Render, Blender Native GPU Module, Bridge Bias, Chang & Luoh, Dust and Cobwebs for Toy Story 4 (SIGGRAPH Talks 2019), Chaotic Cobweb Generator, Clumping, Collision (Bounding Sphere / SDF) (+21 more)

### Community 3 - "Web Mesh Generator"
Cohesion: 0.11
Nodes (24): ARN_OT_generate_web, ARN_WebProps, _build_cobweb(), _build_orb(), build_web_data(), build_web_object(), _env_data(), _finalize() (+16 more)

### Community 4 - "Geometry Node Tree Helpers"
Cohesion: 0.24
Nodes (4): compare_ab(), H, (A, B) input sockets of a Compare node for its current data_type., Helper namespace bound to one node tree.

### Community 5 - "Add-on UI and Registration"
Cohesion: 0.15
Nodes (12): native_available(), native_broken(), enable_gpu_solver(), gpu_backend_available(), Create attributes, add the apply modifier, enable the solver., Panel, ARN_OT_full_setup, ARN_PT_main (+4 more)

### Community 6 - "Native GPU Compute Backend"
Cohesion: 0.18
Nodes (12): apply_arrays(), _bake_sdf(), _collect_colliders(), Mesh objects to collide against: every mesh in the collision     collection if o, Signed distance field of the collider(s), in web-local space,     sampled on a r, Write position/broken/tension arrays into mesh attributes.     Pure CPU — safe t, Read the sim state off the GPU into mesh attributes.         Returns the arrays, _read() (+4 more)

### Community 7 - "Strandify and Dew Baking"
Cohesion: 0.15
Nodes (7): ARN_OT_add_strandify, ARN_OT_bake_dew, ARN_OT_free_dew_bake, Operator, Bake the dew droplet simulation to disk so renders replay it     exactly (the re, Delete the baked dew simulation so it simulates live again     (edit dew setting, Add the strandify modifier (silk tubes + dew) to the active object

### Community 8 - "Viewport Render Screenshot"
Cohesion: 0.31
Nodes (9): Anchor Point Attachment, Sagging Catenary Cross-Strand, Cloth Simulation Relaxation of Web, Cobweb Render Showcase (ss1.png), Dark Viewport Contrast Presentation, Radial Web Sector with Anchor Spokes, Strand Tension Colormap (cyan to red), Strandify Curve Conversion (+1 more)

### Community 9 - "Tearing Solver and Pinning"
Cohesion: 0.22
Nodes (5): ARN_OT_add_tearing_solver, ARN_OT_pin_vertices, Operator, Add the tearing web solver to the active object.     If another mesh is also sel, Write the current Edit Mode vertex selection into the pin attribute

### Community 10 - "Photoreal Cobweb Reference"
Cohesion: 0.48
Nodes (7): Web Anchoring to Surrounding Geometry, Shallow Depth-of-Field Bokeh Backdrop, Dew-Covered Cobweb Reference Photograph, Dew Droplet Beading Along Strands, Irregular Non-Orb Web Topology, Photoreal Reference Target for Generator Output, Strand Catenary Sag Under Gravity

## Ambiguous Edges - Review These
- `Bridge Bias` → `Collision (Bounding Sphere / SDF)`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to
- `Sagging Catenary Cross-Strand` → `Strandify Curve Conversion`  [AMBIGUOUS]
  images/ss1.png · relation: shares_data_with

## Knowledge Gaps
- **2 isolated node(s):** `World-space Gravity and Wind with Turbulence`, `Thomas Kole's Geometry Nodes Cobwebs`
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Bridge Bias` and `Collision (Bounding Sphere / SDF)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Sagging Catenary Cross-Strand` and `Strandify Curve Conversion`?**
  _Edge tagged AMBIGUOUS (relation: shares_data_with) - confidence is low._
- **Why does `H` connect `Geometry Node Tree Helpers` to `GPU Solver Operators`, `Materials and Node Utilities`, `Tearing Solver and Pinning`, `Strandify and Dew Baking`?**
  _High betweenness centrality (0.203) - this node is a cross-community bridge._
- **Why does `NativeState` connect `GPU Solver Operators` to `Native GPU Compute Backend`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Why does `ARN_OT_remove_gpu_solver` connect `GPU Solver Operators` to `Geometry Node Tree Helpers`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `H` (e.g. with `ARN_GPUProps` and `ARN_OT_add_gpu_solver`) actually correct?**
  _`H` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `NativeState` (e.g. with `ARN_GPUProps` and `ARN_OT_add_gpu_solver`) actually correct?**
  _`NativeState` has 5 INFERRED edges - model-reasoned connections that need verification._