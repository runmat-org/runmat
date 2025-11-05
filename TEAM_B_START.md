You're building RunMat. Read the NEXT_PLAN.md file to understand the plan. You are team B. Read the necessary files to understand the codebase and your responsibilities. Keep a record of your progress in a file called TEAM_B_PROGRESS.md. As you work, update the file with your progress, this way if your context is compressed, you can refer to the file to remember what you've done.

# Team B Brief — Runtime, JIT, Fusion Planner & Auto‑Offload

Mission

Make the runtime consistently choose the fastest path with low overhead by introducing a fusion planner, improving auto‑offload policy, integrating epilogues/MRB kernels, and exposing robust telemetry. You own dispatcher integration, fusion planning, native auto‑offload, and results/plots integration.

Context & References

- Primary plan: `NEXT_PLAN.md` (system plan, milestones, KPIs)
- Fusion execution/planning hooks: `crates/runmat-accelerate/src/fusion_exec.rs` (and nearby fusion utilities)
- Auto‑offload model & calibration: `crates/runmat-accelerate/src/native_auto.rs`
- Runtime dispatcher/gather: `crates/runmat-runtime/src/dispatcher.rs`
- Parser/VM context (for correctness interplay): `crates/runmat-parser/src/lib.rs`, `crates/runmat-ignition/src/vm.rs`
- Bench harness: `benchmarks/benchmarks/.harness/` (run_suite.py, suite.json, plotting)

Goals & Success Metrics

- Reduce kernel count for fused pipelines; capture kernel time vs wall time and xfer bytes
- Achieve suite KPIs (see NEXT_PLAN.md) by better planning choices and fusing common DAGs
- Stable, device‑aware offload thresholds with persisted calibration and decision logs

Scope (do)

1) Fusion planner (MVP → extended)
- Pattern‑match DAGs for:
  - PCA centered Gram: `(A; mu=mean(A,1); G=(A−mu)'*(A−mu)/(n−1))` → centered‑GEMM kernel
  - PCA power step normalization: `Q = G*Q; Q = Q ./ (sqrt(sum(Q.^2,1))+eps)` → GEMM+per‑col L2 epilogue
  - Explained variance: `diag(Q' * G * Q)` → GEMM diag/pack epilogue or MRB+pack
- Lower to Team A epilogues/MRB descriptors; ensure residency stays on device

2) Auto‑offload policy
- Persist calibration per device; avoid pathological “no offload” thresholds
- Small‑batch guard: keep CPU for tiny shapes where launch dominates (e.g., 4k B≤8)
- Record offload decisions and thresholds in results JSON for reproducibility

3) JIT integration & warmup
- Per‑case warmup (suite‑driven) rather than global threshold overrides
- Template specialization for kernels (tile @MT@, workgroup @WG@) based on shapes; record selected values

4) Telemetry & results aggregation
- Schema: wall, kernel time, xfer bytes, peak mem, cache hits/misses, device, kernel configs
- Aggregator: compute speedup curves and AUC; produce per‑case rankings and a summary table

Non‑Goals (don’t)

- Author GPU WGSL kernels (Team A responsibility) beyond small glue for planner descriptors
- Change language semantics (Team C handles parser/VM correctness)

Deliverables (tie to milestones in NEXT_PLAN.md)

- M1: Planner MVP for PCA and 4k; telemetry JSON schema; calibrated offload defaults
- M2: Autotune tile/workgroup sizes; persisted calibration; decision logs; results aggregator
- M3: Broader fusion patterns (NLMS, Monte Carlo); graph‑level rewrites; speedup AUC ranking

Design Notes & Constraints

- Planner must conservatively fall back when shapes/dtypes or device features do not match; never regress correctness
- Fusion graph signature should key the fusion cache (DAG + dtypes + shapes) to reuse compiled fused pipelines
- All planning decisions and chosen parameters (e.g., 1‑pass vs 2‑pass, @WG@) should be observable via CLI or recorded in JSON

Testing & Validation

- Planner unit tests for DAG rewrites; golden tests that compare planned vs unplanned graphs numerically (tolerances by dtype)
- Integration: run `run_suite.py` and verify kernel count reduction and speedups; confirm parity_ok stays true
- A/B runs: compare with planner disabled to quantify gains and validate decisions

Execution Checklist

1. Define fusion patterns and descriptors (epilogue masks, MRB reduce dims) and integrate into fusion_exec
2. Wire planner outputs to WGPU provider pathways without host residency changes
3. Implement decision logging and calibration persistence; expose via CLI `accel-info`
4. Update harness to capture telemetry; build aggregator and plots
5. Validate on PCA/4k first; extend to Monte Carlo/NLMS

Definition of Done

- Documented planner with tests; measurable reductions in kernel launches and wall time; parity maintained; telemetry/aggregator outputs produced and checked in.


