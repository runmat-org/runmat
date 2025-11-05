## Team B Shift Handoff

Welcome to the next Team B crew. This document recaps what we inherited (see `TEAM_B_START.md`), what we accomplished (`TEAM_B_PROGRESS.md`), where the shared roadmap sits (`NEXT_PLAN.md`), and what remains most important for you to pick up. It is intentionally detailed—treat it as both a briefing and a ready-reference while you get your bearings.

---

### 1. Context & Mandate Refresh

Team B owns the fusion planner, runtime execution paths, auto-offload policy, telemetry plumbing, and integration glue so RunMat consistently chooses the fastest execution route with minimal overhead. We interface most heavily with:

- **Team A (GPU kernels/provider)** for new fused kernels, precision handling, residency expectations (`crates/runmat-accelerate/…`).
- **Team C (language/runtime semantics)** for parser/VM correctness, RNG semantics, dtype propagation.
- **Team D (telemetry, caches, harness/CI)** for benchmark reporting, cache persistence, and gating.

Our start-of-shift brief (`TEAM_B_START.md`) emphasized three immediate deliverables:
1. Fusion planner MVP for PCA (centered Gram, power-step normalization, explained variance).
2. Auto-offload policy improvements (persisted calibration, small-batch guard, decision logging).
3. Telemetry aggregation + plotting integration for suite-level reporting.

---

### 2. Summary of Work Completed This Shift

Use `TEAM_B_PROGRESS.md` for the line-by-line log. Highlights since takeover:

1. **Fusion Planner & Execution**
   - Implemented planner detection and execution paths for `CenteredGram`, `PowerStepNormalize`, and `ExplainedVariance` patterns.
   - Added `FusionPattern` metadata to carry operands/parameters through planning and execution, keeping constants off the execution stack.
   - Wired the VM (`crates/runmat-ignition/src/vm.rs`) to dispatch the new fusion kinds and restored stack entries on fallback to avoid underflow.
   - Added diverse graph fixtures in `crates/runmat-accelerate/tests/fusion_patterns.rs`, replacing the narrow PCA-only sample.
   - Integration tests in `crates/runmat-ignition/tests/fusion_gpu.rs` now cover CPU↔GPU parity for the new patterns; explained variance matches interpreter behavior by mimicking MATLAB’s column-major transpose quirk.

2. **Runtime Semantics / Constants**
   - Global constants (`NaN`, `Inf`, etc.) are now registered through `runmat-builtins`, allowing fusion tests that rely on omit-nan paths to compile and run without ad-hoc definitions.

3. **WGPU Provider Stabilization**
   - Warmup: Disk cache metadata tracks numeric precision; warmup skips incompatible shaders (fixes M2 Max `entry point 'main'` panic).
   - RNG: Replaced shader-based RNG with host-side generation using the runtime’s 64-bit LCG (`next_uniform_state`). Uniform, normal, integer, and permutation paths now:
     - Generate samples on the host under the provider mutex.
     - Update shared RNG state so interpreter and provider stay in lock-step.
     - Upload results in the provider’s precision (F32/F64). Residency is preserved by uploading into GPU buffers before returning handles.
   - Tests: `rng_wgpu_uniform_matches_cpu` and dependent RNG suites now pass on Apple M2 Max.

4. **Testing Infrastructure**
   - `pause` builtin: Added a test-mode short circuit so unit tests no longer hang waiting for user input.
   - Full `cargo test --features wgpu -- --test-threads=1` runs cleanly **except** for the known BLAS/LAPACK fixture gap (`blas_matmul`, `solve` builtins missing). This matches pre-existing status; see “Open Issues” below.

5. **Documentation & Progress Tracking**
   - Updated `TEAM_B_PROGRESS.md` with detailed milestones to preserve context for this handoff.

---

### 3. Current State Snapshot

- **Tests**
  - `cargo test --features wgpu -- --test-threads=1` passes aside from `crates/runmat-runtime/tests/blas_lapack.rs::{test_builtin_blas_functions, test_builtin_lapack_functions}`. Those fail due to unimplemented builtins (`blas_matmul`, `solve`). All other suites—including fusion, RNG, telemetry, parser, VM—are green.
  - Quick smoke for RNG changes: `cargo test -p runmat-runtime builtins::stats::random::rng::tests::rng_wgpu_uniform_matches_cpu --features wgpu -- --test-threads=1`.

- **Performance Targets**
  - No new benchmarks were run this shift; Team D’s harness remains the source of truth. Telemetry captures upload/download bytes, kernel timings, and fusion hits but still feeds into individual JSONs—suite aggregation isn’t implemented yet.

- **Runtime Behavior**
  - Fusion planner automatically recognizes the PCA pipeline; the VM dispatches fused kernels without manual replays.
  - RNG semantics are single-source: the shared 64-bit LCG drives both host and provider paths, preserving determinism under `rng(seed)`.

---

### 4. Open Issues & Recommended Next Steps

The backlog below is sorted by how directly it impacts Team B’s charter.

1. **Auto-Offload Policy (Top Priority)**
   - Persist calibration per device (Metal vs. CUDA vs. fallback). Current guard avoids `usize::MAX`, but calibration is recomputed every run.
   - Implement the small-batch guard to keep trivial workloads on CPU. Use telemetry to set a heuristic and record decisions.
   - Log decisions and thresholds into results JSON so the suite and UI can explain when/why a workload stayed on CPU.

2. **Telemetry Aggregation & Plotting**
   - Telemetry plumbing is in place (`crates/runmat-accelerate/src/telemetry.rs`, provider instrumentation, CLI hooks), but there’s no suite-level aggregator.
   - Next steps: define JSON schema (maybe piggyback on Team D’s harness), compute speedup curves/AUC, and integrate plotting.

3. **Fusion Planner Extensions**
   - Add detection for MRB/Welford patterns (pushed to later milestones). We have infrastructure to carry metadata; reuse it.
   - Ensure constants/var bindings are preserved when planner is fed graphs from more complex scripts (Team C’s parser fixes may surface new shapes).

4. **BLAS/LAPACK Builtin Coverage (Shared with Team C)**
   - Tests in `crates/runmat-runtime/tests/blas_lapack.rs` expect `blas_matmul` and `solve` builtins. Those are missing; the tests currently fail during full suite runs.
   - Agree with Team C whether to implement these builtins (likely wrappers around existing runtime/accelerate paths) or adjust expectations.

5. **Integration with Team A**
   - SYRK kernel + diag epilogue (Team A) will enable faster centered Gram. Once ready, update planner to route to the new epilogue instead of the current generic path.
   - When Team A exposes fused MRB kernels, add corresponding `FusionKind` and `FusionPattern` entries.

6. **Documentation & CI**
   - Fold RNG behavior into developer docs (note the deterministic host-side generation and state sync).
   - Work with Team D to add CI gating for RNG parity (quick test) and fusion pattern detection to catch regressions.

---

### 5. How to Resume: Practical Checklist

1. **Environment**
   - `rustup` nightly toolchain pinned in repo rust-toolchain file.
   - Apple M2 Max requires `RUSTFLAGS="-C target-cpu=native"` for best numbers when benchmarking.

2. **Validation** (run in order after pulling latest)
   ```bash
   cargo fmt
   cargo clippy --workspace --all-targets --features wgpu
   cargo test --features wgpu -- --test-threads=1  # expect BLAS/LAPACK failure until builtins land
   ```
   - Optional quick checks: `cargo test -p runmat-runtime builtins::array::creation::rand --features wgpu`, `cargo test -p runmat-accelerate tests::fusion_patterns`.

3. **Benchmark Prep**
   - Team D’s harness lives under `benchmarks/.harness`. After implementing auto-offload updates, re-run `python run_suite.py --device auto` to gather new telemetry.

4. **Coding Guidance**
   - Use `FusionPattern` metadata for any new planner rule; avoid pushing constants onto the execution stack.
   - When touching RNG or telemetry, update both host/runtime and provider sides to keep behavior consistent.
   - For auto-offload logging, coordinate with `runmat/src/main.rs` so CLI users can inspect decisions.

---

### 6. Coordination Notes

- **Team A**
  - Awaiting kernels: SYRK, MRB/Welford, diag epilogue. Once landed, expect interface changes in `AccelProvider` (new hooks). Keep planner ready to consume `FusionPattern::CenteredGram { normalization }` metadata.

- **Team C**
  - RNG semantics are now aligned; ensure future parser/VM changes preserve Value IDs to keep planner variable binding intact.
  - BLAS/LAPACK builtin expectations need coordination (test suite currently fails).

- **Team D**
  - Telemetry aggregator remains outstanding; align on schema before emitting data. The CLI currently dumps raw telemetry snapshots.

---

### 7. Risks & Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Auto-offload changes regress performance on small workloads | Suite speedups drop; CI could catch late | Implement decision logging & guard rails; test both CPU/GPU paths for representative sizes before landing |
| Fusion planner mis-detects patterns when constants/vars change | Incorrect fusion or execution failure | Extend unit fixtures with more varied graphs (Team C to supply), keep `FusionGroupPlan::const_values` handling tight |
| RNG changes introduce performance regression for very large tensors | Additional host→device copies | Monitor telemetry upload bytes; consider chunked generation if runtime shows hot spots |
| BLAS/LAPACK builtins absent | Current test failure blocks full CI once enforced | Align with Teams A/C on implementing wrappers or deferring tests |

---

### 8. Reference Index

- `TEAM_B_START.md` — initial mandate, goals, outstanding items.
- `TEAM_B_PROGRESS.md` — chronological log of actions.
- `NEXT_PLAN.md` — cross-team roadmap, milestone definitions, KPIs.
- Key source directories:
  - Planner/execution: `crates/runmat-accelerate/src/fusion.rs`, `fusion_exec.rs`.
  - VM integration: `crates/runmat-ignition/src/vm.rs`.
  - RNG + provider: `crates/runmat-runtime/src/builtins/common/random.rs`, `crates/runmat-accelerate/src/backend/wgpu/provider_impl.rs`.
  - Tests: `crates/runmat-accelerate/tests/fusion_patterns.rs`, `crates/runmat-ignition/tests/fusion_gpu.rs`, RNG tests under `crates/runmat-runtime/src/builtins/array/creation/`.

---

### 9. Closing Thoughts

You’re inheriting a runtime that now auto-fuses the PCA pipeline, keeps RNG parity across CPU/GPU, and has the telemetry scaffolding ready for aggregation. The next phase is more strategic: align auto-offload heuristics with real telemetry, wire suite-level reporting, and continue widening fusion coverage as Team A delivers kernels. If you need a quick mental model: **planner detects → plan metadata carries constants cleanly → VM dispatches → provider executes with residency preserved**. Keep that pipeline tight and we’ll stay ahead of NumPy/Torch.

Good luck, and feel free to reach out if anything in this hand-off needs clarification.

— Team B (Previous Shift)

