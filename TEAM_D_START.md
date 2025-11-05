## Team D Kickoff Brief — Optimization Caches, Telemetry, Benchmarks & CI

Welcome to Team D. This playbook summarizes the mandate handed down in `NEXT_PLAN.md`, captures the current state of the system, and outlines the immediate priorities so you can execute quickly. Treat it as the living “start here” document—update it as you learn more.

---

### 1. Mission & Scope

**Mandate (from `NEXT_PLAN.md`):** Consolidate optimization caches, telemetry, reproducible benchmarks, and CI gates to sustain speed and quality. Team D owns the glue between provider/runtime instrumentation and the tooling that proves regressions aren’t creeping back in.

You are responsible for:

- Cross-run caches (pipelines, bind groups, buffers, fusion groups, moments/statistics) and their invalidation/versioning.
- Telemetry schema definition, transport, aggregation, and user-facing reporting (CLI + plots).
- Benchmark harnesses and their automation, including producing comparative results across NumPy/Torch and RunMat.
- CI gating that enforces both correctness and performance targets.

You coordinate most closely with Team A (who supplies kernels/pipelines), Team B (planner/runtime integration and telemetry emitters), and Team C (language/runtime correctness that benchmarks exercise).

---

### 2. Current State Snapshot (inheritance summary)

This shift hands you the following baseline (see `TEAM_B_PROGRESS.md` & `TEAM_B_OUT.md`):

1. **Provider / Runtime Telemetry**
   - `crates/runmat-accelerate/src/telemetry.rs` defines counters (kernel time, uploads/downloads, cache hits).
   - WGPU provider records upload/download bytes, fused dispatch timings, matmul timings. CLI exposes raw snapshots via `runmat accel-info`.
   - No suite-level aggregator exists; telemetry is emitted per run but not summarized.

2. **Caches & Warmup Infrastructure**
   - Pipeline cache persistence is in place with versioning and numeric precision awareness. Warmup skips incompatible shader blobs.
   - Buffer pooling (`provider_impl.rs`) exists with basic size bins, but no eviction policy or telemetry.
   - No generalized bind group layout cache, fusion group cache, or moments cache yet.

3. **Benchmarking Harness**
   - Python harness under `benchmarks/.harness` supports param sweeps, env injection, and device reporting.
   - Results today are per-run JSON artifacts; there is no aggregator computing speedup curves/AUC, nor plot automation integrated into CI.

4. **CI / Automation**
   - CI does not gate on performance metrics. Some correctness suites run, but there are no thresholds for speedup regression or telemetry validation.
   - RNG parity, fusion detection, and warmup caches are tested manually at the moment.

5. **Documentation & Telemetry Policy**
   - `docs/TELEMETRY.md` covers installer telemetry only; runtime telemetry is undocumented beyond code comments.

Takeaway: instrumentation plumbing exists, but aggregation, caching sophistication, and CI enforcement have not started.

---

### 3. Immediate Priorities (First 1–2 sprints)

Align with the R1 deliverables in `NEXT_PLAN.md`, translated into actionable tasks:

1. **Define & Emit Telemetry Schema**
   - Establish a canonical JSON schema for per-run telemetry (wall time, kernel time, transfers, peak mem, cache hits/misses, device, kernel configs, fusion hits).
   - Wire Team B’s telemetry provider to emit this schema; ensure backward compatibility with CLI commands.

2. **Suite-Level Aggregator & Visualization**
   - Extend `benchmarks/.harness` to ingest telemetry, compute speedup curves and AUC against NumPy/Torch baselines, and generate plots (likely via `runmat-plot` or Matplotlib).
   - Produce a single “results package” artifact per suite run (JSON + PNG/SVG).

3. **Cache Inventory & Gaps**
   - Audit existing caches (pipeline, buffer, warmup). Document lifetime, invalidation, metrics.
   - Prototype bind-group layout cache keyed by layout signatures, with telemetry hooks to track hits/misses.
   - Draft designs for fusion-group cache and moments/statistics cache, informed by Team B’s fusion patterns.

4. **CI Integration Plan**
   - Define which telemetry metrics become CI gates (e.g., merged speedup AUC thresholds, pipeline cache hit rate minima).
   - Sketch pipeline: harness run → artifact ingest → threshold comparison → GH status.

Deliverable: a README or design doc summarizing schema, aggregator architecture, and cache roadmap.

---

### 4. Secondary Objectives (Next Milestones)

Once the immediate tasks are stable:

- Implement reduction-plan cache and moments cache to reuse mean/ex² buffers across runs.
- Persist heuristic hints per device (auto-offload thresholds, workgroup sizes) fed by telemetry.
- Add RNG/tile caches for tiny workloads (ring-buffer concept from `NEXT_PLAN.md`).
- Hook aggregator output into documentation/website to surface public speedup dashboards.
- End-to-end CI: nightly suite with published telemetry, gating on parity and performance.

---

### 5. Coordination Checklist

| Team | What you need | How to engage |
| --- | --- | --- |
| **Team A** | Pipeline metadata (tile sizes, shader IDs), cache hooks, new kernel configs | Weekly sync; request new telemetry fields before kernel merges |
| **Team B** | Emission points in runtime, planner hints for fusion caches, auto-offload logs | Pair on schema integration; ensure logging doesn’t regress planner state |
| **Team C** | Benchmark correctness baselines; RNG semantics for distribution tests | Share harness expectations; align on test seeds and dataset storage |
| **Infra (CI)** | Access to GH Actions, artifact storage, secrets management | File infra tickets early if new storage or runners needed |

---

### 6. Working Environment & Commands

1. **Build/Test sanity**
   ```bash
   cargo fmt
   cargo clippy --workspace --all-targets --features wgpu
   cargo test --features wgpu -- --test-threads=1
   ```
   _Note:_ BLAS/LAPACK tests currently fail due to missing builtins; coordinate with Teams B/C before enabling in CI.

2. **Harness usage**
   ```bash
   cd benchmarks/.harness
   python run_suite.py --device auto --output ../results/<timestamp>
   ```
   Add flags/env variables (see harness README) for param sweeps or warmup toggles.

3. **Telemetry inspection**
   ```bash
   runmat accel-info --format json --reset
   ```
   Returns current provider counters—use during schema validation.

4. **WGPU pipeline cache**
   Cached binaries live under `~/.runmat/pipelines`. Precision mismatches are skipped automatically after recent fixes.

---

### 7. Risks & Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Telemetry schema churn breaks CLI or harness | Users lose visibility / CI false negatives | Version the schema, provide migration tooling, keep backward compatibility toggles |
| Aggregator becomes a bottleneck (slow data crunch) | CI runtimes blow up | Streamline pipeline (incremental stats, limit data volume), parallelize or sample |
| Cache persistence causes stale or corrupted pipelines | Incorrect kernels deployed | Version caches aggressively, store integrity checks, expose manual reset commands |
| Performance gating is noisy (flaky baselines) | CI instability | Use controlled hardware, multiple iterations, statistical thresholds rather than single-run equality |

---

### 8. References & Further Reading

- `NEXT_PLAN.md` — Team D workstreams and milestone KPIs (R1–R3).
- `TEAM_B_OUT.md` — Notes on telemetry gaps and expectations from Team B.
- `benchmarks/.harness/README.md` — Harness usage and configuration.
- `docs/TELEMETRY.md` — Installer telemetry policy; extend this for runtime telemetry once schema is finalized.
- WGPU pipeline cache implementation: `crates/runmat-accelerate/src/backend/wgpu/cache/`.
- Provider telemetry source: `crates/runmat-accelerate/src/backend/wgpu/provider_impl.rs` (search `record_*`).

---

### 9. Initial Todo (recommended first PRs)

1. Draft telemetry schema proposal (RFC or ADR) and circulate with Teams A/B/D.
2. Instrument harness to ingest current telemetry JSON, even if schema is provisional.
3. Build a minimal aggregator that emits speedup tables and total kernel time summaries.
4. Document cache behavior and expose CLI commands for inspection/reset.

Update this document as you close these items or learn new constraints. Good luck—Team D sits at the nexus of data and discipline; your work keeps the rest of RunMat honest.

