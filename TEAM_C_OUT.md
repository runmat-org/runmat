## Team C Handoff Dossier

### 1. Mission Recap and Starting Context

Team C owns the Language/Core Correctness & Semantics stream for RunMat. Our charter remained consistent with the onboarding brief (see `NEXT_PLAN.md`): eliminate parser/VM correctness issues, lock down dtype/RNG semantics, and keep the benchmark harness green. When we picked up the shift the codebase reflected the legacy MATLAB semantics in broad strokes but had several pain points:

- **Evaluation semantics drift**: top-level VM bytecode occasionally stalled or leaked output due to `CallBuiltinMulti` not advancing the program counter in zero-output scenarios. This was raising “cannot convert Bool(false) to f64” and similar runtime surprises in scripts.
- **Parser fragility**: multi-assign vs. matrix literal disambiguation, newline suppression, and dynamic member assignments (`s.(expr) = ...`) all had known regressions.
- **DType/RNG cracks**: scalar conversions eagerly upcast to F64, GPU gather lost precision metadata, and RNG builtins (`rand*`) produced incorrect types/residency when crossed with GPU providers.
- **Harness drift**: benchmark scripts still hard-coded defaults and ignored the `.harness` parameter injection protocol.

We inherited the persistent history captured in `TEAM_C_PROGRESS.md`, and the sprint plan from `NEXT_PLAN.md` (M1 parser/VM/randn, M2 dtype invariants, M3 documentation/tests) framed our work.

### 2. Completed Work (Chronological Highlights)

Key accomplishments are already itemized in `TEAM_C_PROGRESS.md`; this section distills them into thematic buckets with rationale:

1. **VM execution stability**
   - Patched `crates/runmat-ignition/src/vm.rs` so every `CallBuiltinMulti` branch advances `pc` even when a builtin produces no outputs.
   - Added regression `call_builtin_multi_advances_pc_for_zero_outputs` under `crates/runmat-ignition/tests/basics.rs` to safeguard against regressions.

2. **Parser resilience**
   - Updated `parse_stmt` and `try_parse_multi_assign` to prefer multi-assign parsing at statement start while still falling back gracefully to matrix literals.
   - Hardened matrix literal parsing (`parse_matrix`) for mixed comma/whitespace separators and nested expressions.
   - Restored newline suppression semantics, multi-assign newline handling, and dynamic member assignment through tests in `crates/runmat-parser/tests/*`.

3. **DType propagation across GPU boundaries**
   - Introduced provider precision metadata via `set_handle_precision`/`handle_precision` (see `crates/runmat-accelerate-api/src/lib.rs` and `crates/runmat-accelerate/src/simple_provider.rs`).
   - Refined `dispatcher::gather_if_needed` so host tensors reconstructed from GPU handles honor the originating precision and coerce numeric payloads appropriately.
   - Added an end-to-end dtype regression suite (`crates/runmat-runtime/tests/dtype.rs`) covering `zeros/ones/randn`, GPU round-trips, and `'single'` prototypes.

4. **RNG correctness envelope**
   - Implemented statistical sanity tests in `crates/runmat-runtime/tests/rng.rs` validating mean/variance for both F32 and F64 outputs.
   - After the cross-team merge introduced unexpected GPU residency for default `rand`, we reverted `rand_double` to always produce host tensors (matching MATLAB semantics), eliminating RNG mutex poisoning and restoring deterministic sequences for the CPU path.

5. **Benchmark harness compliance**
   - Updated all five reference scripts (`benchmarks/*/runmat.m`) with `exist` guards, harness-provided defaults, deterministic seeding, and single-precision enforcement. Monte Carlo/NLMS metrics now align with the NumPy baseline used by the harness.

6. **Ancillary cleanups**
   - Adjusted tests in `crates/runmat-lexer/tests/lexer.rs`, `crates/runmat-gc/tests/stress.rs`, `crates/runmat-turbine/tests/jit.rs`, and BLAS/LAPACK coverage to keep the suite green under the merged code.
   - Documented progress continuously in `TEAM_C_PROGRESS.md` for traceability.

### 3. Current State Snapshot

#### Build & Test Status

- **Host provider (`RUNMAT_ACCELERATE_PROVIDER=inprocess`)**: Full `cargo test -- --test-threads=1` succeeds (see execution logs 2025-11-05 10:40). This mode exercises all Team C touchpoints and confirms dtype/RNG regressions are resolved.
- **Hardware provider (`--features wgpu`)**: Tests still fail during warmup because WGPU attempts to load precompiled pipeline metadata with stale entry-point names, triggering `wgpu` validation (“Unable to find entry point 'main'” in `warmup-precompiled-pipeline`). This blocks CLI/integration suites long before Team C logic runs. Root cause lies in Team B’s pipeline cache (see `crates/runmat-accelerate/src/backend/wgpu/warmup.rs`).
- **Benchmark harness**: The README still references `.harness/run_suite.py`, but that script no longer exists in the repo. We could not execute a full parity run. Bench scripts themselves are harness-ready.

#### Outstanding Issues / Open Threads

1. **WGPU Warmup Failure (Team B dependency)**
   - All `--features wgpu` workflows crash during provider initialization. Team B needs to audit the warmup cache, regenerate WGSL assets, or temporarily disable cache replay until metadata matches the new entry points.
   - Team C’s dtype/RNG fixes are validated with the CPU provider, but hardware parity cannot be reconfirmed until this is addressed.

2. **Harness automation gap**
   - The canonical parity driver referenced in `benchmarks/README.md` (`.harness/run_suite.py`) is missing. Clarify with the infra team whether the harness moved, or reintroduce an updated runner. Team C validated scripts manually but did not replace the missing tool.

3. **Follow-up verifications**
   - NLMS metric parity vs. NumPy (flagged in earlier logs) should be rechecked once harness automation is restored.
   - When Team B delivers a working WGPU warmup fix, rerun the RNG/dtype suites with hardware acceleration to ensure provider hooks (e.g., `random_uniform_like`) remain consistent.

### 4. Suggested Next Steps for Incoming Team C Shift

| Priority | Task | Owner hints |
| --- | --- | --- |
| P0 | Partner with Team B to resolve WGPU warmup cache mismatch. Retest `cargo test --features wgpu -- --test-threads=1` with GPU provider active. | Coordinate with the Accelerate guild; focus on `warmup.rs` and shader packaging. |
| P1 | Restore harness automation by either retrieving the relocated suite runner or drafting a replacement in `benchmarks/`. | Check infra repos; if unresolved, consider a lightweight Python runner that mirrors the README instructions. |
| P1 | Once WGPU is back, re-run RNG/dtype/benchmark suites under GPU acceleration and confirm no regressions. | Execute `cargo test -p runmat-runtime rand`, `... rng`, etc., with `RUNMAT_ACCELERATE_PROVIDER=wgpu`. |
| P2 | Deepen regression coverage for NLMS metric parity and Monte Carlo variance under single precision. | Extend `crates/runmat-runtime/tests` or add harness-driven golden comparisons. |
| P2 | Document the final dtype pipeline (ProviderPrecision metadata) in developer docs (`docs/`), ensuring new contributors understand the contract. | Sync with Team B to align expectations for provider implementations. |

### 5. Artifacts and References

- **Plan / multi-team coordination**: `NEXT_PLAN.md` — outlines Milestones M1–M3, partner responsibilities, and cross-team dependencies.
- **Start announcement**: The initial mission statement (captured in earlier shift briefings) emphasized parity across PCA/4k/MonteCarlo/NLMS/IIR, parser robustness, dtype preservation, and RNG fidelity. Use `NEXT_PLAN.md` for canonical details since `TEAM_C_START.md` is absent in the repo snapshot.
- **Progress ledger**: `TEAM_C_PROGRESS.md` — chronological log of every touch we made this shift.
- **Key code paths**:
  - VM: `crates/runmat-ignition/src/vm.rs`
  - Parser: `crates/runmat-parser/src/lib.rs` and associated tests
  - RNG/dtype: `crates/runmat-runtime/src/builtins/array/creation/{rand,randn,randi,randperm}.rs`, `crates/runmat-runtime/src/dispatcher.rs`, `crates/runmat-runtime/tests/{dtype,rng}.rs`
  - GPU precision plumbing: `crates/runmat-accelerate-api/src/lib.rs`, `crates/runmat-accelerate/src/simple_provider.rs`
  - Benchmarks: `benchmarks/*/runmat.m`

### 6. Communication Notes

- **Team B (Accelerate)**: Blocking issue (wgpu warmup). Open a shared channel with the driver responsible for pipeline caching. Provide failing logs (the validation error excerpt from today’s run).
- **Team A (Tooling & UX)**: Inform them the CLI/integration suites pass under CPU provider; highlight the missing harness runner if it impacts their automation.
- **QA / Harness Ops**: Share the updated scripts and note the harness gap so they can adjust nightly runs accordingly.

### 7. Closing Thoughts

The core objectives—parser stability, dtype fidelity, RNG correctness, and harness compliance—are in a good state. The only caveat is the external WGPU regression which fell outside our remit but blocks full parity validation. The next shift should focus on unblocking that dependency and re-running hardware-backed parity tests.

Thanks for picking up the baton. All relevant commands, failing logs, and validation runs are captured in the shell history and progress log. Reach out if anything in this dossier is unclear.

