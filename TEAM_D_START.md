### 1. Mission & Scope

**READ THIS FILE: NEXT_PLAN.md** We are Team D.

## Current Priorities (in ship-blocking order)

1. **Bind-group cache + buffer pool tuning (R1)**  
   - *Why now*: Every unfused dispatch currently rebinds layouts and churns allocations, showing up as overhead in PCA/4K case runs. This blocks us from hitting the headline “RunMat is faster than NumPy/Torch” metrics.  
   - *Plan*: Implement keyed bind-group caches with dynamic offset support, expand buffer pools with size classes and eviction, and add telemetry to track hit/miss ratios.  
   - *Dependencies*: None; enables Team A’s launch reuse path and stabilizes warmup behavior.

2. **Fusion group cache MVP (R1)**  
   - *Why now*: Team B’s planner emits fusion patterns, but we currently regenerate WGSL pipelines per run. Caching compiled kernels keyed on fusion DAG + dtype is required to keep repeated workloads fast and makes the warmup story credible.  
   - *Plan*: Ship an in-memory cache (with hooks for later persistence) plus telemetry for cache efficiency.  
   - *Dependencies*: Builds on bind-group telemetry to quantify benefits; otherwise self-contained.

3. **Cross-run pipeline warmup & persistence (R2)**  
   - *Why now*: Launch messaging relies on fast first-run experiences. Persisted warmups let us match or beat NumPy/Torch on one-shot demos.  
   - *Plan*: Capture warmed pipelines per shape/device, tie into existing pipeline cache versioning, surface CLI/ENV controls for prewarming common kernels.  
   - *Dependencies*: Requires stable bind/fusion caches to avoid reusing stale state.

4. **Reduction-plan & moments caches (R2)**  
   - *Why now*: PCA and 4K flows rely on repeated mean/variance computations. Without cached planning decisions, the planner overhead offsets kernel wins.  
   - *Plan*: Cache Welford/MRB planning metadata and per-handle mean/ex² tensors, with hit/miss telemetry.  
   - *Dependencies*: Coordination with Team B’s planner to share signatures.

5. **Persisted heuristic hints + CI performance gates (R3)**  
   - *Why now*: Once caches land, we need device-tuned thresholds and automated gating so regressions can’t slip in pre-release. This is the last mile to confidently claim “fastest math runtime”.  
   - *Plan*: Store per-device auto-offload and workgroup hints, record metadata in suite summaries, wire CI to compare normalized AUC against reference baselines on our M2/RTX targets.  
   - *Dependencies*: Relies on suite aggregator (already landed) and reference hardware telemetry.

6. **Telemetry CLI/report polish (R3)**  
   - *Why now*: Important for developer visibility but not a launch blocker. Schedule after performance gates are in place.  
   - *Plan*: Extend CLI to render the aggregated telemetry and speedup summaries; document workflow in `docs/TELEMETRY.md`.
