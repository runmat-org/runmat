You're building RunMat. Read the NEXT_PLAN.md file to understand the plan. You are team C. Read the necessary files to understand the codebase and your responsibilities. Keep a record of your progress in a file called TEAM_C_PROGRESS.md. As you work, update the file with your progress, this way if your context is compressed, you can refer to the file to remember what you've done.

# Team C Brief — Language/Core Correctness & Semantics

Mission

Eliminate parser/VM correctness issues, finalize dtype and RNG semantics, and ensure scripts/harnesses run cleanly so performance work is meaningful and reproducible. You own parser, VM execution semantics, runtime builtins (zeros/ones/randn/gpuArray/gather), and benchmark script correctness.

Context & References

- Primary plan: `NEXT_PLAN.md` (system plan, per‑case fixes)
- Parser: `crates/runmat-parser/src/lib.rs`
- VM: `crates/runmat-ignition/src/vm.rs`
- Builtins & dtype: `crates/runmat-builtins/src/lib.rs`, `crates/runmat-runtime/src/builtins/**` (zeros/ones/randn), `crates/runmat-runtime/src/dispatcher.rs` (gather)
- GPU dtype inference with gpuArray/like: `crates/runmat-runtime/src/builtins/acceleration/gpu/gpuArray.rs`
- Bench scripts and harness: `benchmarks/benchmarks/*`, `.harness/*`

Goals & Success Metrics

- Suite “parity_ok” true across PCA, 4k, Monte Carlo, NLMS, IIR for small/medium profiles
- No VM/Parser runtime errors (e.g., “cannot convert Bool(false) to f64”, “SliceNonTensor”, assignment context errors)
- DType (F32/F64) preserved across gpuArray/zeros/ones/randn/gather; ‘single’ round‑trip validated
- RNG correctness: randn(mean≈0,var≈1) in both F32/F64; provider precision respected

Scope (do)

1) VM top‑level evaluation semantics
- Ensure top‑level control flow / non‑numeric expressions do not produce values coerced to f64
- Confirm `Pop`/`CallBuiltinMulti` paths properly advance PC and suppress unintended output

2) Parser robustness
- Keep newline as statement separator; stabilize multi‑assign and lvalue forms; improve command‑form disambiguation
- Add tests for tricky constructs (e.g., `[a,b]=f(); A(1)=x; s.f=x; s.(n)=x`) and ensure semicolon suppression behavior matches MATLAB

3) DType and builtins
- Ensure `zeros/ones(...,'single')`, `randn(...,'single')`, and `... like(proto)` produce correct `NumericDType` and shape
- Preserve dtype in `gather` from `provider.precision()`; verify scalar conversions do not upcast unexpectedly

4) Scripts & harness conformance
- Make PCA/4k/NLMS/IIR use guarded defaults so harness‑injected assignments take effect: `if ~exist('n','var'), ... end` (etc.)
- Remove host round‑trip hacks from scripts when provider capabilities suffice; keep two‑output econ‑QR to avoid slicing
- Align NLMS metric with NumPy reference; verify MSE magnitude parity

Non‑Goals (don’t)

- Author GPU kernels (Team A) or fusion planner (Team B), beyond correctness enabling changes

Deliverables (tie to milestones in NEXT_PLAN.md)

- M1: VM top‑level suppression fix; parser stability tests; randn('single') validated; scripts adopt harness params; parity_ok across cases
- M2: DType invariants and gather semantics validated; remove remaining script workarounds; additional language coverage as needed by benchmarks
- M3: Documentation and examples; CI regression tests for parser/VM edge cases

Testing & Validation

- Unit tests: parser (token/AST), VM (pc increments, expression suppression), dtype conversions, randn distributions (F32/F64)
- Integration: run `run_suite.py`; verify no runtime errors and parity_ok
- Spot checks: PCA explained variance consistency; 4k MSE parity; Monte Carlo price within 1%; NLMS MSE parity

Execution Checklist

1. Implement VM suppression logic; add tests to cover previous failure signatures
2. Harden parser for newline/multi‑assign/lvalue; add regression tests
3. Validate dtype flows across zeros/ones/randn/gpuArray/gather; fix any mismatches
4. Clean scripts to honor harness assignments; align metrics and seeds; remove debug
5. Run suite; ensure parity_ok; hand off any perf gaps to Team A/B

Definition of Done

- Suite runs without parser/VM/builtin errors; parity_ok true; dtype semantics correct; scripts harness‑compliant; changes documented.


