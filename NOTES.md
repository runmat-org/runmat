# q-hero-pack notes

Helper q-hero-pack (base t1-transplant 11baf9d1de). Website P0: battery-pack hero.
Work dir: /Users/nallana/.claude/jobs/b8482b6b/tmp/work/q-hero-pack (pack copy, scripts, study, renders).
Build Mac: ~/src/q-hero-pack, ~/targets/q-hero-pack, ~/runs/q-hero-pack, logs ~/logs/qhp-*.

## 1. Pack STEP import
- Cause: `admit_measured_edge_tolerances` (occt_bridge.cc) raises an understated edge tolerance to
  the measured curve/pcurve gap only when the gap is <= 10x the written tolerance. The pack writes
  0.9 nm on every edge; real gaps are 8-13 nm, so they were left and the incidence check failed.
- Fix: admit gaps up to max(10x written, 1 um absolute). (A 10 nm floor in the core check was tried
  and dropped: the pack has 12.7 nm gaps.)
- Next failure: pcurve u = 1 + 3e-9 past UResolution-padded face bounds (1 + 1.2e-9). Fix in
  exact_surface_evaluator.cc: clamp an overshoot whose surface point is within 1.1x the face's
  max edge tolerance of the clamped point.
- Incidence refinement now follows the gap function (curve flatness bisected nm-tolerance edges
  into um intervals: >8 min); surface lookup per pcurve sample indexed; unrestricted adaptor.
- Result: pack imports in 52 s (survey test, debug build of test, OCCT release), 0.8 GB RSS;
  CLI `geometry.load` 57.6 s, 2.0 GB RSS (runmat-hero-943c19ab1e). Max gap 23 nm.
- Survey harness: crates/runmat-geometry/io/tests/hero_pack_import.rs (ignored; RUNMAT_LARGE_STEP).
- Regression: external_cad_corpus.rs exported_pack_cable_with_nanometre_pcurve_gaps_imports.

## 2. Study
- Module STEP derived with work/q-hero-pack/scripts/make_module.py (solid names kept; 39 coolant
  channel faces named "M01 coolant channel wall"). Study: work/q-hero-pack/study/m01-module-thermal.fea.

## 3. Rendering
- view(az, el) used azimuth from +x; MATLAB measures from -y. Fixed in Camera::view_direction_deg;
  figures without a view call now start at MATLAB's default view(-37.5, 30).
- 3-D plot axes now start at the lower corner of the axis limits and tick those limits
  (AxisFrame3d, shared by renderer and overlay); CAD geometry keeps the world-origin triad.
- axis equal / daspect no longer stretch 3-D limits into a cube (flat pack filled ~30% of frame);
  the axis scale carries the aspect (data_aspect_axis_scale).
- Test: crates/runmat-plot/tests/three_d_framing.rs (RUNMAT_PLOT_FRAMING_OUT keeps PNGs).
- Tests: runmat-plot lib 320/0; runtime plotting 1308/0; geometry core 93/0, io 123/0 + integration.

## Module mesh
- 8 mm run on hero-943c19ab1e failed after 37 min (Delaunay insertion invariant). q-mesh-fix-6
  fix 830c112746 cherry-picked (ec4a5a07fa); the mesh-fix helper verifies on its own CLI.
