## Team A Handoff – GPU Kernels & WGPU Provider (UTC 2025-11-06 22:15)

### Mission Summary
- **Goal:** Keep RunMat’s WGPU backend ahead of NumPy/PyTorch by delivering high-performance kernels, data movement optimizations, and telemetry.
- **Current Focus:** Finish Team A milestones (vec4 GEMM/SYRK, logical transpose, telemetry) on the path to fused 4k pipelines and reliable benchmarks.

### Work Completed This Shift
- Delivered logical-transpose support without kernel launches:
  - Added transpose metadata tracking (`TransposeInfo`) in `runmat-accelerate-api`.
  - Updated `transpose_exec` to create zero-copy views and taught matmul/matmul_epilogue/pagefun to honour `MATMUL_FLAG_TRANSPOSE_*` flags.
  - Adjusted pool reuse and download paths to respect shared-buffer lifetimes and column-major materialisation.
  - Added WGSL shader support for transpose-aware tiled and small-k GEMM kernels.
  - Added regression tests (`transpose_roundtrip_matches_cpu`, `matmul_with_transposed_operand_matches_cpu`).
- Kept vec4-only kernels and planner integration stable (ran `cargo test -p runmat-accelerate --features wgpu transpose`).

### Repo State & Key Artifacts
- **API:** `crates/runmat-accelerate-api/src/lib.rs` – new transpose metadata helpers.
- **Provider:** `crates/runmat-accelerate/src/backend/wgpu/provider_impl.rs` – logical transpose implementation, matrix view helpers, buffer pooling guard.
- **Shaders:** `crates/runmat-accelerate/src/backend/wgpu/shaders/matmul.rs`, `shaders/matmul_smallk.rs` – transpose flags for WGSL kernels.
- **Params:** `crates/runmat-accelerate/src/backend/wgpu/params.rs` – `MatmulParams.flags` and constants.
- **Tests:** `crates/runmat-accelerate/tests/transpose.rs` – new coverage. Existing vec4 tests still green.
- **Docs:** `TEAM_A_PROGRESS.md` updated through this handoff (timestamp 22:05 UTC).

### Outstanding Work (Team A Charter)
1. **team-a-2 – Vec4 load/store paths (In Progress):**
   - Vec4 matmul shipped (F32 lanes, alignment guard). SYRK vec4 path exists but needs revalidation after transpose changes.
   - TODOs: benchmarking vs scalar path, heuristics for non-4 multiples, ensuring planner/runtime can request vec4 safely.
2. **team-a-3 – Logical transpose support (Completed this shift).**
3. **team-a-4 – Telemetry extensions (Pending):**
   - Emit counters for vec4/transposed pathways (kernel duration, bytes moved, pool hits/misses).
   - Wire new metrics into benchmark harness (`benchmarks/.harness/run_suite.py`).

### Recommended Next Steps
1. **Validate Vec4 SYRK under logical transpose:**
   - Re-run `cargo test -p runmat-accelerate --features wgpu syrk` and add a vec4-specific regression mirroring matmul.
   - Benchmark covariance/PCA workloads to confirm throughput gains.
2. **Deliver telemetry (team-a-4):**
   - Extend `crates/runmat-accelerate/src/telemetry.rs` with vec4/transpose counters and expose via provider API.
   - Update harness scripts to record the new metrics for CI dashboards.
3. **Planner/runtime alignment:**
   - Coordinate with Team B so fusion emits logical transposes instead of materialised buffers.
   - Audit exec paths that still assume physical transposes (e.g., covariance) and update as needed.
4. **Performance/Regression Sweep:**
   - Bench 4k image pipeline, PCA/covariance, GEMM microbench (vec4 vs scalar).
   - Monitor buffer pool usage now that multiple handles can share underlying storage.
5. **Documentation & Coverage:**
   - Annotate transpose helpers (2-D only today) and add F32 transpose round-trip tests.

### Risks & Mitigations
- **Buffer Pool Saturation:** Logical views keep base buffers alive; pooling now checks `Arc::strong_count`, but monitor telemetry for capacity pressure.
- **Planner Expectations:** Older fusion paths may still materialize transposes—sync with Team B to avoid redundant copies.
- **Telemetry Gap:** Until team-a-4 lands, vec4/transpose success is opaque. Prioritize instrumentation next shift.

### Useful Commands
- `cargo test -p runmat-accelerate --features wgpu transpose`
- `cargo test -p runmat-accelerate --features wgpu syrk`
- `RUNMAT_WGPU_FORCE_PRECISION=f32 cargo test -p runmat-accelerate --features wgpu matmul_small_k`

### Coordination
- Team B (planner) for transpose metadata propagation and future fusion patterns.
- Team C (benchmarks) for integrating telemetry counters.
- See `NEXT_PLAN.md` for KPI targets (4k pipeline P95 latency, GEMM parity vs CuBLAS baseline).

### Closing
Logical transpose is now zero-copy and ready for planner-driven fusion. Next shift should stabilise vec4 SYRK, stand up telemetry, and keep the fusion roadmap aligned with these runtime primitives.