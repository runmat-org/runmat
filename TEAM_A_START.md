You're building RunMat. Read the NEXT_PLAN.md file to understand the plan. You are team A. Read the necessary files to understand the codebase and your responsibilities. Keep a record of your progress in a file called TEAM_A_PROGRESS.md. As you work, update the file with your progress, this way if your context is compressed, you can refer to the file to remember what you've done.

# Team A Brief — GPU Kernels & Provider (WGPU)

Mission

Build the fastest GPU execution path for RunMat across core math workloads by implementing specialized kernels, fusing epilogues, and minimizing launch/transfer overhead. You own the WGPU provider, shader kernels, residency, pipeline/bindgroup caches, and buffer lifecycle.

Context & References

- Primary plan: `NEXT_PLAN.md` (consolidated system plan and milestones)
- Codebase touchpoints:
  - WGPU provider: `crates/runmat-accelerate/src/backend/wgpu/provider_impl.rs`
  - WGSL kernels (matmul, reductions, etc.): `crates/runmat-accelerate/src/backend/wgpu/shaders/`
  - Tiled matmul: `crates/runmat-accelerate/src/backend/wgpu/shaders/matmul.rs`
  - Symmetric ops placeholder: `crates/runmat-accelerate/src/backend/wgpu/shaders/symmetry.rs`
  - Provider init/defaults: `runmat/src/config.rs` (WGPU default)
  - Auto‑offload model & calibration guards: `crates/runmat-accelerate/src/native_auto.rs`
  - Fusion execution hooks (to coordinate with Team B): `crates/runmat-accelerate/src/fusion_exec.rs`
  - Bench harness: `benchmarks/benchmarks/.harness/` (run_suite.py, run_bench.py, suite.json)

Goals & Success Metrics

- PCA: ≥2× NumPy end‑to‑end for n∈[10k,80k], d∈{512,1024}, k=8, parity mode (fp32)
- 4k: ≥1.3× NumPy at B∈{4,8,16}; ≥2× at B≥32
- Monte Carlo: ≥2× NumPy for M∈[250k,1M], T=256
- NLMS/IIR: ≥1.5× NumPy across profiled sizes
- No correctness regressions; suite “parity_ok” remains true

Scope (do)

1) Specialized linear algebra
- SYRK for G=Aᵀ·A (triangular update), k‑chunking for large n
- Small‑k optimized GEMM (e.g., 32×8/16×8 tiles for k≤8), unrolled inner loops
- Logical transpose support via strides (avoid materializing Aᵀ)

2) GEMM epilogues (fused)
- Epilogue A (elementwise): scale, bias, clamp, pow
- Epilogue B (per‑axis): column/row norms and standardize (map→reduce→broadcast within tile)
- Epilogue C (diag/pack): compute/store diagonals/selected outputs without extra kernels

3) Fusion & overhead
- 4k pipeline: fuse normalize→gain→bias→clamp→pow into 1–2 kernels
- Monte Carlo: fuse drift+scale·Z→exp→multiply; reuse buffers each step
- NLMS: fuse y/e/nx/update per step; reduce kernel count
- Launch reduction: bind‑group reuse, dynamic offsets, minimal command re‑recording

4) IO, precision, and memory
- Vectorized IO (vec4) and coalesced access in kernels (matmul, SYRK, MRB)
- Extend buffer pooling (size classes; zero‑on‑alloc for zeros/ones)
- Centralized allocation preflight (`create_storage_buffer_checked`) enforced in hot paths
- Respect `NumericDType` (F32/F64); prefer F32 on devices lacking f64 shaders

Non‑Goals (don’t)

- Large algorithmic changes outside epilogues/fusion families (e.g., altering PCA algorithm semantics)
- CPU provider optimization (except necessary fallback correctness)

Deliverables (tie to milestones in NEXT_PLAN.md)

- M1: Epilogue A/B for GEMM; fused 4k kernel; vec4 paths in matmul/SYRK; improved buffer pooling
- M2: Centered‑GEMM v1 (on‑the‑fly centering + epilogue scale) and MRB kernel patterns where applicable
- M3: Small‑k GEMM specialization; diag/pack epilogues; Monte Carlo fused step; NLMS fused update; logical transpose by strides
- M4: Caches/warmup: bind‑group cache, fusion group cache contributions; moments cache integration with MRB; telemetry surfaced via CLI
- M5: Graph‑level fusion support with Team B’s planner (keep kernels ready/configurable)

Design Notes & Constraints

- Pipeline keys MUST include: shader bytes, layout tag, workgroup size (@WG@), precision, epilogue mask, reduce‑dims mask, constants mask
- Keep kernels parameterized (tile @MT@, workgroup @WG@) and expose env overrides for tuning
- Avoid device↔host transfers inside hot loops; enable slicing or specialized pack kernels to replace host slicing
- Use shared memory prudently; cap temporary state to avoid register/spill issues; gate epilogues by size heuristics

Testing & Validation

- Unit tests (Rust): numerical parity for epilogues (norms/scales), diag/pack vs host; SYRK vs matmul reference; small‑k vs general GEMM
- Microbenchmarks: `G=Aᵀ·A` (gonly), `GQ` small‑k; measure tile/vec4 and epilogue benefits
- Integration: PCA/4k/NLMS/Monte Carlo small+default; ensure no extra gathers; record kernel count and xfer bytes

Telemetry & Reporting

- Expose provider counters: fused cache hits/misses, warmup duration, kernel time, transfer bytes, peak mem
- Persist pipeline caches; log key/tile/workgroup settings for reproducibility
- Emit device info (backend, name) into results JSON (already partially wired)

Execution Checklist (repeatable)

1. Build with WGPU features: `cargo build -p runmat --release -F runmat-accelerate/wgpu -F runmat-runtime/wgpu`
2. Implement kernel/epilogue; update pipeline keys and provider wiring
3. Add unit tests and microbench(s)
4. Run `benchmarks/benchmarks/.harness/run_suite.py` (JSON + plots)
5. Record speedups and parity; iterate tile/@WG@ if needed
6. Land docs in kernel headers and `NEXT_PLAN.md` delta notes

Definition of Done

- Bench suite shows targeted speedups with parity; telemetry recorded; caches warmed; no regressions in other cases; code documented and tested.


