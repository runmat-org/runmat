# Merge Stabilization Log

- 2025-11-07 00:00 UTC — Kicked off merge stabilization. Plan: fix WGPU handle registration regression, rerun accelerated tests, investigate `runmat-syrk-vec4-shader` validation, then verify harness + telemetry.
- 2025-11-07 00:05 UTC — Replaced obsolete `set_handle_transposed` call in `register_existing_buffer` with `clear_handle_transpose` to restore compatibility with the new transpose metadata API.
- 2025-11-07 00:34 UTC — `cargo test -p runmat-accelerate --features wgpu -- --test-threads=1` now passes; confirmed fused ImageNormalize, SYRK vec4, and transpose suites are green post-fix.
- 2025-11-07 00:54 UTC — Reproduced earlier release build scenario with `cargo run -p runmat --release -F runmat-accelerate/wgpu -F runmat-runtime/wgpu -- run benchmarks/4k-image-processing/runmat.m`; run completed without WGPU validation errors (may have been resolved by recent merges or stale cache issue).
- 2025-11-07 01:05 UTC — Harness `pca` smoke surfaced `accel-info` error (`wgpu feature not enabled`); determined release binary built without the crate-level `runmat` `wgpu` feature even though dependencies used WGPU. Need to rebuild binary with `-F wgpu` so telemetry collection works.
- 2025-11-07 01:18 UTC — Rebuilt `target/release/runmat` with `-F wgpu -F runmat-runtime/wgpu`; `runmat accel-info --json` now returns full telemetry payload.
- 2025-11-07 01:26 UTC — PCA harness still fails (`Index 1 out of bounds (1 to 0)` during diag/explained variance) even with WGPU binary; repro confirmed via direct script run. Needs follow-up from runtime/fusion teams.
- 2025-11-07 01:30 UTC — 4k image-processing harness succeeds on WGPU with MSE=0 and telemetry snapshot (though counters remain zero), confirming fused ImageNormalize path executes end-to-end.
- 2025-11-07 01:38 UTC — Entered follow-up phase: triaging PCA fusion failure (`Index 1 out of bounds`), empty telemetry counters, and full-suite validation. Added TODO tracking (todo-4..6).
- 2025-11-07 01:45 UTC — Added verbose instrumentation in `execute_explained_variance` gated by `RUNMAT_DEBUG_EXPLAINED` to capture Q/G shapes and intermediate results while debugging PCA failure.
- 2025-11-07 02:25 UTC — Identified PCA failure root-cause: `exist('n','var')` pre-creates workspace entries (value 0) so harness guard never overwrote defaults. Reworked ignition workspace tracking to distinguish assigned variables (new `assigned` set, propagated via `push_pending_workspace`/`take_updated_workspace_state`) so `exist` respects uninitialized slots.
- 2025-11-07 02:58 UTC — Repaired REPL semicolon regression: expression results now store in dedicated temp slots and workspace tracking handles control-flow assignments, restoring `semicolon_suppression` suite to green.
- 2025-11-07 03:12 UTC — PCA harness (`target/release/runmat benchmarks/pca/runmat.m`) now completes via WGPU provider with sane explained variance output (`RESULT_ok`) confirming workspace fix resolves prior out-of-bounds.
- 2025-11-07 03:28 UTC — WGPU provider now records matmul/fused-elementwise dispatch telemetry; added regression test and harness summaries (fusion+bind group cache totals) to surface the counters.
- 2025-11-07 03:45 UTC — `RUNMAT_TELEMETRY_OUT` piping implemented; harness consumes JSON snapshots with non-zero upload/matmul/fused counters for PCA & 4k runs (bind-group/fusion cache totals included).

