## RunMat Acceleration: System Plan to Consistently Outperform NumPy/PyTorch

## Executive summary

RunMat is close to GPU-accelerated parity on a subset of math workloads, but gaps remain in both correctness and performance versus NumPy/PyTorch. Recent work fixed major blockers (WGPU default enablement, GPU buffer pooling and checks, dtype plumbing for single, randn('single'), parser/VM stability, shared-memory tiled matmul, auto-offload calibration sanity). The benchmark suite now runs parametrically with device reporting and plots. However, we still see:

- PCA: Parity mismatches (metrics/logs), slowdowns from host round-trips and CPU QR; missed specialized kernels (SYRK), no fused diag.
- 4k image processing: Correctness aligned; small-batch performance is launch-bound and unfused; wins appear at larger batch sizes only.
- Monte Carlo: Correctness historically off (now addressed with randn('single')), but throughput limited by many small kernels and RNG speed.
- Batched NLMS/IIR: Metric misalignment and VM top-level output issues; per-iteration unfused kernels.

Goal: Within three execution rounds, deliver a system that consistently outruns NumPy and PyTorch for a broad class of math workloads on laptop and workstation GPUs/CPUs, with correctness parity, reproducible benchmarking, and CI gates to prevent regressions. We organize this effort across three teams with crisp interfaces and shared telemetry.

## Current state and findings (from suite)

- Infrastructure
  - WGPU provider now default (runmat/src/config.rs → AccelerateConfig.provider=Wgpu); Metal backend on Apple Silicon; f64 shader unavailability handled via F32 kernels.
  - Prebuild binary for suite; eliminate build time from runs; device reported per-impl.
  - Harness supports param sweeps via env and HARNESS_ASSIGN injection for RunMat; warmup supported; plotting improved.
  - Auto-offload thresholds: guarded against ‘usize::MAX’ calibrations (native_auto.rs); still needs policy tuning.

- Correctness
  - 4k MSE parity: aligned across NumPy/Octave/Torch; RunMat differed due to clamp/pow and keepdims; fixed.
  - Monte Carlo: randn('single') implemented (crates/runmat-runtime/src/builtins/array/creation/randn.rs); price parity target within 1%.
  - PCA: re-orthonormalization aligned to econ-QR, but slicing on GPU requires host round-trip. Need to ensure harness params apply unambiguously.
  - VM/parser: multi-assign, newline handling improved (crates/runmat-parser/src/lib.rs), but top-level non-numeric printing still surfaces in some scripts; needs a VM fix.

- Performance
  - PCA G=A'·A: tiled shared-memory matmul added (matmul.rs), k-chunk accumulation for large k; performance improved but still unfused epilogues and CPU QR.
  - 4k: Launch overhead dominates small B; only shows wins at larger B; pipeline not fused.
  - Monte Carlo/NLMS/IIR: many small kernels; no fusion; RNG throughput not optimized; resource reuse limited by per-iter allocations.

## Team structure and mandates

### Team A: GPU Kernels & Provider (WGPU)

Mandate: Make the GPU backend definitively faster than NumPy/Torch across core math primitives by implementing specialized kernels, fusing common patterns, and reducing launch/transfer overhead. Own: crates/runmat-accelerate (backend/wgpu), shaders, residency, pipeline cache, memory mgmt.

#### Key workstreams

1) Specialized Linear Algebra & Epilogues
   - SYRK for G=A'·A: Implement a symmetric rank-k update kernel for covariance-like patterns; exploit symmetry to halve work and bandwidth.
   - Small-k GEMM path: Optimize G·Q when k≪d with column-blocking and shared-memory tiles sized for k.
   - Fused diag(Q'·G·Q): Epilogue in GEMM/SYRK to accumulate only diagonal of Q'·G·Q without materializing intermediates.
   - Logical transpose support: Stride remapping to avoid materializing transposes for @' and .' operations.

2) Kernel Fusion & Graph Execution
   - 4k pipeline fusion: Combine normalize (mean/var), affine (gain/bias), clamp, pow into 1–2 kernels; adopt one-pass variance (Welford) to halve global memory passes.
   - Monte Carlo fusion: Combine drift+scale·Z → exp → multiply in a single kernel per step; reuse buffers.
   - NLMS fusion: Fuse y, e, nx, and W update in one kernel; precompute reductions per column efficiently.

3) Launch/Transfer Overhead Reduction
   - Bind group reuse and dynamic offsets for frequently reused buffers.
   - Command re-recording minimization; pipeline cache versioning (already added) and pre-warm tuning.
   - Residency and pooling: Extend buffer pooling (provider_impl.rs) with size class bins and LRU; zero-on-alloc semantics for zeros/ones; stricter bounds checks (create_storage_buffer_checked).

4) Precision/IO Optimization
   - Everywhere vectorized IO (vec4 loads/stores) and coalescing; tune MATMUL_TILE and workgroup sizes.
   - Enforce NumericDType end-to-end; prefer F32 on GPUs lacking f64; expose dtype in ‘like’ flows.

#### Deliverables (Team A)

- R1: SYRK kernel + fused diag epilogue; 4k fused kernel (normalize→pow); buffer pooling improvements; vec4 IO in matmul/syrk. KPI: PCA small wins ≥2.0× over NumPy at n∈[10k,80k]; 4k B∈[4,8,16] ≥1.3× over NumPy.
- R2: Small-k optimized GEMM; Monte Carlo fused kernel; NLMS fused kernel; logical transpose by strides; launch reuse. KPI: Monte Carlo ≥2× NumPy; NLMS ≥1.5× NumPy.
- R3: Graph-level fusion for common patterns; autotune tile sizes per device; on-disk pipeline cache across runs. KPI: Maintain ≥1.5× median speedup across suite profiles; no regression.

#### Fusion architecture (consolidated)

- GEMM epilogue family: A) elementwise map (scale/bias/clamp/pow), B) per-axis reduce+scale (norms/standardize), C) diag/pack (selective write-back). Implement as epilogue descriptors; include in pipeline keys.
- Centered-GEMM: mean-subtract-on-load and `(1/(n−1))` apply in epilogue; variant 2 fuses running moments.
- MRB generator: map→reduce→broadcast kernels for mean/std/var, L2 normalize; broadcast/reduction planning caches.
- Fusion graph rewrites: detect PCA centered Gram, PCA power-step normalization, explained-variance diag pack; avoid host residency changes.

### Team B: Runtime, JIT, Fusion Planner & Auto-Offload

Mandate: Ensure the runtime always chooses the fastest path with low overhead, fuses ops where possible, and provides telemetry and CI gates. Own: runmat runtime (dispatcher, fusion planner), Turbine JIT, native_auto thresholds.

#### Key workstreams

1) Fusion Planner & Shape-Aware Paths
   - Introduce a simple fusion IR that recognizes PCA and 4k patterns; lower to Team A epilogues.
   - One-pass reductions: teach planner to pick Welford-based kernels when requested metrics allow.

2) Auto-Offload Policy
   - Device-aware thresholds (Metal vs CUDA) with persisted calibration; avoid pathological calibrations (already guarded).
   - Small-batch guard: keep CPU for tiny workloads where GPU launch overhead dominates; record decision in results JSON.

3) JIT Integration & Warmup
   - Warmup per case (suite-driven) instead of lowering global thresholds; capture JIT compile vs run time.
   - Specialize kernels (template params: tile sizes, vec widths) at runtime for shapes.

4) Telemetry & Reproducibility
   - Record: wall, kernel time, xfer bytes, peak mem, cache hits/misses; device info; kernel configs (tile sizes, WG, precision).
   - Results aggregator: compute speedup curves, AUC, and rank implementations.

#### Deliverables (Team B)

- R1: Fusion planner MVP (PCA diag fusion, 4k fused path); calibrated offload defaults; suite telemetry JSON schema. KPI: Reduced kernel count (4k halves) and demonstrated speedups.
- R2: Autotune tile sizes; persisted calibration per device; decision logs; JIT compile amortization evidence. KPI: Additional 10–20% across matmul-heavy cases.
- R3: Broaden fusion patterns (NLMS, Monte Carlo); integrate graph execution path. KPI: ≥1.5× median suite speedup sustained.

### Team C: Language/Core Correctness & Semantics

Mandate: Eliminate parser/VM correctness issues, finalize dtype and RNG semantics, and stabilize behavior for benchmarking and user scripts. Own: parser (crates/runmat-parser), VM (crates/runmat-ignition), runtime builtins and dtype plumbing.

#### Key workstreams

1) Parser & VM Robustness
   - Top-level non-numeric suppression: ensure statements that evaluate to logicals/void do not print/coerce to f64; fix IIR/L harness failures.
   - Finalize multi-assign, newline as separator, lvalue assignment parsing (already improved).

2) DType & Prototype Semantics
   - NumericDType completed; ensure ‘single’ flows through gpuArray/zeros/ones/randn and gather; preserve dtype on device→host.
   - ‘like’ semantics: infer dtype from prototype (Value::GpuTensor queries provider.precision()).

3) RNG and Randomness
   - rand, randn exact options for 'single' and 'like'; provider-aware precision; validate distribution stats in tests.

4) Scripts & Harness
   - Ensure PCA/4k/NLMS/IIR adopt harness params via guarded defaults; unify seeds; remove debug paths.

#### Deliverables (Team C)

### Team D: Optimization Caches, Telemetry, Benchmarks & CI

Mandate: Consolidate optimization caches, telemetry, reproducible benchmarks, and CI gates to sustain speed/quality. Own: provider and fusion caches, warmup profiles, harness plotting and result schema, CI workflows.

#### Key workstreams

- Pipeline/module caches: stable hashing, on-disk persistence, versioning; warmup hints and selective pre-warm.
- Bind group/layout caches: key by (pipeline, layout signature); offset-aware variants for chunked dispatch.
- Buffer lifecycle: size-classed pools for STORAGE/COPY; scratch workspaces; growth/decay heuristics.
- Broadcast/reduction planning caches: fast-path scalar ops, one-pass vs two-pass per device; persist thresholds.
- Fusion group cache: DAG signature (ops+dtypes+shapes) → WGSL + pipeline IDs; constants baked via uniforms where beneficial.
- Moments/statistics cache: cache mean/ex2 on handles with dims mask; reuse across MRB.
- RNG state/streams: device RNG with host-synced seed; ring-buffer tiles for tiny shapes.
- Telemetry schema: wall, kernel time, xfer bytes, peak mem, cache hits/misses, device, kernel configs; exposed via CLI.
- CI/bench: small profiles with parity gates (rtol/atol) and speedup thresholds; reproducible seeds; optional fixed inputs.

#### Deliverables (Team D)

- R1: Bind group + buffer pools; basic fusion group cache; telemetry schema surfaced; warmed frequent pipelines.
- R2: Reduction plan cache; moments cache; cross-run pipeline warmup; results aggregator with AUC speedup and ranking.
- R3: Persisted heuristic hints per device; CLI reporting; CI gating of parity/speed.

- R1: VM top-level suppression; remaining parser edge cases; finalized randn('single') tests. KPI: suite parity_ok across all cases.
- R2: DType propagation tests and runtime invariants; thorough gather semantics; correctness harness for corner cases. KPI: stable behavior across devices.
- R3: Expanded language coverage (as needed by benchmarks); documentation.

## Per-case plans

PCA
- Correctness: Guard defaults in runmat.m; single-precision end-to-end; sanity log TRACE_G=sum(diag(G)).
- Performance: Team A implements SYRK and fused diag; Team B fuses epilogue via planner; Team A/B avoid host round-trips by enabling GPU slicing or building a narrow ‘take-first-k-columns’ kernel; autotune matmul tiles for d=512.

4k Image Processing
- Correctness: Maintain keepdims parity; clamp-before-pow.
- Performance: Fused kernel; Welford variance; small-B CPU guard in auto-offload; measure kernel launches and xfer bytes as KPIs.

Monte Carlo Analysis
- Correctness: randn('single'); price parity within 1% across seeds.
- Performance: Fused step; GPU RNG; buffer reuse; CPU/GPU decision by M×T size; vec4 loads.

Batched NLMS
- Correctness: Align MSE metric with NumPy; add small test to suite.
- Performance: Fuse y/e/nx/update per step; reuse temporaries; reduce kernel count significantly.

Batched IIR Smoothing
- Correctness: VM fix; verify mean parity; guarded defaults for M/T.
- Performance: Single-kernel loop with persistent Y; CPU guard at tiny M.

## Cross-cutting: tooling, CI, and fairness

- Device policy: RunMat=WGPU default; Torch auto (CUDA>MPS>CPU); record device; do not force downlevels.
- Prebuild RunMat; warmup runs before timing; no global JIT threshold overrides unless per-case justified.
- Telemetry schema: write kernel-time, transfers, peak mem, fused-cache hits; store kernel configuration (tile sizes, vec widths).
- CI Gating: small profile per case with speedup thresholds vs fixed baselines; parity gates on metrics (rtol/atol).
- Reproducibility: seed control; optional fixed-input datasets (e.g., PCA A, 4k images) for parity checks.

## KPIs and targets

- PCA: ≥2× NumPy across n∈[10k,80k] on Apple M2 Max; maintain correctness.
- 4k: ≥1.3× NumPy at B∈[4,8,16], ≥2× at B≥32; one-pass variance and fused kernel.
- Monte Carlo: ≥2× NumPy for M∈[250k,1M] and T=256; price within 1%.
- NLMS: ≥1.5× NumPy across C∈[512,4096] for fixed p,T; metric parity.
- IIR: ≥1.5× NumPy for M∈[250k,1M]; correctness first.
- Suite AUC speedup: ≥1.5× median speedup across all profiled points.

## Risks and mitigations

- GPU slicing/host round-trips: Implement minimal ‘slice-first-k-cols’ kernel or enable slicing semantics on GpuTensor to avoid gather.
- Kernel correctness when fusing: Add micro-tests and compare against high-precision references; use tolerances appropriate to F32.
- Auto-offload mis-picks: Persist calibration; log decisions; expose override env for experiments.
- MPS/CUDA feature gaps (e.g., linalg.qr on MPS): Record device and expected fallbacks; ensure fair comparisons (report-only, do not force CPU).

## Milestones

M1 — Epilogue foundation and 4k fusion
- GEMM epilogue A/B; planner emits GEMM+epilogue for PCA normalization; 4k normalize→pow fused; vec4 IO in matmul/SYRK; buffer pooling tuned.

M2 — Centered‑GEMM and reduction/MRB
- Centered‑GEMM with on‑the‑fly centering; MRB generator for mean/std/var/L2; reduction planning cache; device-aware offload defaults.

M3 — Small‑k GEMM, diag/pack epilogues, Monte Carlo/NLMS fusion
- Small‑k GEMM specialization; diag/pack epilogues; Monte Carlo fused step with GPU RNG; NLMS fused update; logical transpose by strides.

M4 — Caches, warmup, and telemetry
- Fusion group/bind group caches; moments cache; cross-run warmup; CLI telemetry; results aggregator with speedup AUC and ranking.

M5 — Graph-level fusion and CI gating
- Graph-level rewrites for common patterns; persisted calibration/hints; CI parity and performance gates; documentation and examples.

## Appendix: relevant code touchpoints

- Provider init/defaults: runmat/src/config.rs (AccelerateConfig.provider=Wgpu); runmat/src/main.rs initializes with config.
- WGPU backend/provider: crates/runmat-accelerate/src/backend/wgpu/provider_impl.rs (buffer pooling, create_storage_buffer_checked, matmul_exec, rng); shaders/matmul.rs (tiled); shaders/symmetry.rs (symmetric ops, to be extended for SYRK).
- Auto-offload: crates/runmat-accelerate/src/native_auto.rs (calibration guarding, thresholds).
- DType plumbing: crates/runmat-builtins/src/lib.rs (NumericDType); runtime builtins zeros/ones/randn single support; dispatcher gather preserves dtype.
- Parser/VM: crates/runmat-parser/src/lib.rs (newline, lvalue/multi-assign, command-form guards); crates/runmat-ignition/src/vm.rs (pc increment for qr; top-level suppression fix pending).
- Harness: benchmarks/benchmarks/.harness (run_bench.py, run_suite.py, suite.json); plots and device logging.

## Appendix: env & config knobs (fairness)

- RunMat Accel: RUNMAT_ACCEL_ENABLE, RUNMAT_ACCEL_PROVIDER=wgpu|inprocess, RUNMAT_ACCEL_WGPU, RUSTMAT_ACCEL_WGPU_POWER, RUSTMAT_ACCEL_DISABLE_FALLBACK.
- JIT: RUSTMAT_JIT_ENABLE/THRESHOLD/OPT_LEVEL (no suite-wide overrides by default; use warmup).
- Case params: PCA_N/D/K/ITERS; IMG_B/H/W; MC_M/T; NLMS_P/C/T; IIR_M/T.

## Closing

This plan aligns kernel specialization and fusion (Team A), runtime policy and fusion planning (Team B), and correctness/semantics (Team C) to systematically exceed NumPy/Torch. With clear KPIs, telemetry, and CI gating, we can sustain speedups and avoid regressions while preserving MATLAB/Octave semantics.