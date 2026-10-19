# Mesher and realistic FEA studies: engineering handoff

**Page 1 of 2 | 18 October 2026 | Work in progress, not release-qualified**

## Branch, purpose, and completion target

The main repository is **runmat-analysis**, branch **continue-meshing-work**. Its pre-handoff HEAD is `ed062b92f47432bd20f55a90f4ce5f281dfa3d10`. Companion guide/graphics work lives in **runmat-private**, branch **geometry-work** (pre-handoff HEAD `7945babd8cd09e80e92c966c9f47c3dd515951c4`). The new handoff commits preserve the working source, including the latest unfinished repair. They are not qualification or release commits.

The immediate objective is to finish two realistic, reproducible guides at the quality of the existing linear-static guide: an aluminium heat sink with approximately 10 W base heating and convection, retaining temperature and heat-flow fields; and a tuning fork with a fixed stem and free tines, retaining two natural frequencies and mode shapes. The shipped sandbox **bracket.stp**, using bare CLI defaults, is also a required mesher regression. Completion includes accepted meshes and solves, original saved results, cold reopening, six inspected figures, complete downloadable projects, and working website delivery.

These models exposed general defects in meshing, resource accounting, transport, and scheduling. The work therefore repairs the shared implementation instead of simplifying geometry, increasing tolerances, expanding resource ceilings, or maintaining guide-only numerical paths. Guide source lives under `docs/simulation/guides/`; examples and CAD live under `docs/simulation/examples/`.

## What has been accomplished

The branch contains substantial geometry/meshing and FEA work: machine-aware CPU admission and reusable execution resources; bounded surface serialization and cold materialization; compact indexed transport; incremental topology/quality state and local transactions; directional sizing with bounded noncommuting source combinations; surface support consistent with volume metrics; explicit integration/state contracts; and modal output/resource profiling. Requirements, witnesses, and docs export are retained alongside the implementation. Broader physics/release qualification remains open.

The last frozen source inventory, **1256f358c86e** (a content identity, not a Git commit), passed **1,284 owner tests and all 25 native/resource/CLI gates**. This covered native CAD, meshing, transport, resource regressions, requirements export, and representative cold publication. Those results apply to that snapshot, not automatically to the latest scheduling draft.

## Actual model status and measured failure

**No current original study is accepted: 0/2 studies and 0/2 delivered guides.** The latest bracket passed surface joining and initial volume construction, but final quality convergence hit its original one-hour task budget: **3,676.61 s** total, exit 1, confirmed quiescence, sampled summed physical peak **1.76 GB**. The external memory/time guard did not fire. The previous surface-join memory rejection did not recur; this attempt failed on convergence/work cost. Fork and heat were not launched after that failure. Historically, the fork meshed but modal resource preparation rejected its output estimate; heat exposed excessive sizing/representation costs. Fixes have regression evidence but still require fresh original-study acceptance.

A **60-second replay of the exact retained bracket quality input** exposed refinement-frontier starvation: hundreds of insertion trials rejected by the joint size/shape policy, with no optimizer pass reached. A one-second diagnostic sample placed 441/659 samples in refinement; this is a local observation, not an hour-long CPU profile. The optimizer's admitted billion-candidate ceiling is real but was not the first measured bottleneck. Waiting for the full-hour verdict was unnecessarily expensive; future iteration must start at this retained numerical boundary.

<!-- PAGEBREAK -->

**Page 2 of 2 | Repair state, evidence, and execution plan**

## Latest repair: exactly what is and is not verified

The latest draft lets refinement try its existing source window, then yield to coupled repair rather than scan the whole mesh first. A repair rejection remains provisional: refinement may still unlock another legal move. Accepted state changes invalidate stale rejection/cursor information. Final topology and both numerical quality criteria remain mandatory; declared budgets and cancellation must still be enforced.

The changed numerical files are relative to `crates/runmat-meshing/tetrahedron/src/cdt/`: `volume_refinement/working.rs`, `volume_mesh/quality_convergence.rs`, and the extracted `volume_mesh/quality_convergence/repair.rs`. The original candidate-budget regression is unchanged. An earlier shape-first draft passed 569 tests but failed that budget test; it was revised rather than relaxing the test. **The current handoff revision passes library/test type checking only. Its full tests, retained-input convergence, and performance are unverified.** No original retry is justified yet.

## Fast, evidence-driven path to completion

1. **Close the scheduling repair.** Add adversarial coverage for failed source windows yielding to repair, rejected repair resuming refinement, stale-state invalidation, true global exhaustion, budgets, cancellation, and work independent of unrelated mesh inventory. Keep ownership in the numerical coordinator/refinement modules; do not add guide-specific fallback solvers.
2. **Use the exact retained input as the local gate.** Build only the changed owner and its existing `volume_quality_replay` example. Run short, question-driven diagnostics with direct events. Require actual convergence and bounded work before another whole-model attempt. A diagnostic process exiting zero merely reports its verdict; `diagnostic-stopped-not-acceptance` is never a numerical pass. Measure performance with optimized code; a faster development build is for correctness iteration only.
3. **Freeze and integrate once ready.** Run owner tests/lint, architecture, resource/native/CLI gates, then unchanged bracket, fork, and heat cases. Reuse accepted upstream artifacts only when their identities remain valid. Avoid a full release build prerequisite, duplicate compiler lanes, and unchanged failed reruns.
4. **Finish the actual deliverables.** Check thermal residual and approximately 10 W energy balance; modal eigen-residuals and complete original basis. Validate cold results and graphics replay, inspect all six native posters, publish through the shared docs export, and verify website assets and downloaded ZIP inventories. Do not claim mesh-convergence or production qualification from solve residuals alone.

## Inputs, evidence, and working commands

Heat retains Tet4, 10 mm target, 0.1 mm deviation, conductivity 160 W/(m K), convection 35 W/(m2 K), ambient 298.15 K, and 2,290 W/m2 on the original heater face. Fork retains Tet10, 30 mm target, 0.1 mm deviation, steel E=200 GPa, nu=0.3, density 7,850 kg/m3, fixed stem base, and two modes. Keep authored 512 MiB/one-hour and external 8 GiB/two-hour limits; build/test guards are separate. Preserve the bracket's CLI defaults.

A separately transferable, Git-ignored bundle under `.handoff/` contains the 15,952,044-byte quality input, its exact controls, SHA manifest, frozen verification receipts, original failure, bounded replay log/sample, and local orchestration references. Input SHA256: `15b18f2b002962415acec9b531e240329de66f05ecc5061db1c4acb1c707a424`. Original receipts remain in `/private/tmp/runmat-applied-guide-models/` on this machine. Bundled coordinators contain local paths and source fences; rebind deliberately before use.

From the analysis repository, use Rust **1.90.0**, `--offline -j1`, and the canonical target directory. Start with `cargo +1.90.0 test -p runmat-meshing-tetrahedron --lib --offline -j1`; the existing replay example accepts `INPUT.cbor CONTROLS.json SECONDS` (1-300). Local OpenTelemetry is available through `otell intro` and `otell search`. Preserve Zed/terminals and the sister `runmat` build. GPU expansion is deferred. Transfer the companion private commit and evidence bundle together with this branch; do not mistake the handoff commit for completed studies.
