## Team B Handoff — Runtime, JIT, Fusion Planner & Auto-Offload

Welcome, next crew! Below is the state of Team B as of this handoff. It captures what we shipped, what’s still hot, and how to get moving quickly without stepping on other teams.

---

### 1. Mission Refresher

Team B owns the runtime planner layer: fusion detection/execution, auto-offload policy, JIT warmup and telemetry/reporting glue. The goal remains: ensure RunMat always picks the fastest path with clear instrumentation and reproducibility.

Key repos/components we touch:
- `crates/runmat-accelerate/src/fusion.rs`, `fusion_exec.rs` — fusion detection + executor plumbing.
- `crates/runmat-accelerate/src/native_auto.rs` — auto-offload policy, calibration, telemetry.
- `benchmarks/.harness/run_suite.py` — benchmark aggregator/telemetry summariser.
- `runmat/src/main.rs` — CLI entry points exposing provider info/calibration.

---

### 2. What We Accomplished This Shift

**Fusion / Execution**
- Added `FusionKind::ImageNormalize` detection, metadata (`ImageNormalizePattern`, `ImageScalar`) and executor hook that drives Team A’s fused WGSL kernel. No provider edits required.
- Updated fusion planner fixtures/unit tests and added `image_normalize_matches_cpu` integration test using a seeded sample + test provider implementation of the pipeline.

**Auto-Offload Policy & Calibration**
- Extended suite telemetry (`_summarize_runmat_telemetry`) to emit `auto_offload_calibration` summaries capturing CPU time, unit counts, provider info, etc.
- Added runtime ingestion: `apply_auto_offload_calibration_from_file()` reads suite JSON, recomputes CPU cost coefficients, persists optional cache updates, and stores before/after deltas.
- Surfaced the history through `AutoOffloadReport` (previous thresholds + delta) and exposed the new data via `runmat accel-info`.
- Added CLI command `runmat accel-calibrate --input <suite_results.json> [--dry-run] [--json]` to apply/preview calibrations standalone.

**Documentation / Tracking**
- Updated `TEAM_B_PROGRESS.md` with details of the fusion and calibration workstreams for future reference.

---

### 3. Current State & Validation

**Tests run**
- `cargo test -p runmat-accelerate --test fusion_patterns`
- `cargo test -p runmat-ignition --test fusion_gpu`

**CLI smoke**
- `runmat accel-info` (with and without `--json`) now shows the calibration summary.
- `runmat accel-calibrate --dry-run --json --input <suite_results.json>` previews coefficient updates; omit `--dry-run` to persist the cache.

No outstanding lint/clippy errors introduced by this shift.

---

### 4. Next Priorities (Team B backlogged items)

1. **Harness / Telemetry Enhancements (next-in-queue)**
   - Extend `_summarize_runmat_telemetry` to capture new provider counters once Team A lands them (e.g. image-normalize hits, vec4 branch usage, logical transpose). The groundwork for calibration aggregation is already in place—reuse the same pattern.
   - Emit fused-path flags in the per-case summaries so plotting/reporting can highlight where fusion kicked in.
   - Coordinate with Team D to ensure the new summary schema is compatible with their plotting scripts; update `plot_suite.py` and downstream dashboards once counters land.

2. **Auto-Offload Policy Iteration**
   - With calibrations now ingestible, schedule PCA/4k suite sweeps on each tier device (M2 Max, workstation GPU) and commit updated thresholds to cache + repo.
   - Verify CLI output against expectations (before/after + delta) and ensure we document the calibration workflow for others.

3. **Fusion Planner Roadmap**
   - Coordinate with Team A regarding upcoming fused kernels (vec4-friendly matmul epilogues, logical-transpose). Once APIs land, add planner detection + executor wiring similar to the ImageNormalize flow.

---

### 5. How to Resume

**Environment**
- Standard `rustup` toolchain pinned via `rust-toolchain.toml`.
- GPU validation done on Apple M2 Max with WGPU enabled; use `RUSTFLAGS="-C target-cpu=native"` for local perf tests.

**Kick-off Checklist**
1. Pull latest + run `cargo fmt`, `cargo test -p runmat-accelerate --test fusion_patterns`, `cargo test -p runmat-ignition --test fusion_gpu` to confirm baseline.
2. Run `runmat accel-info --json` to ensure the calibration summary appears; optionally try `runmat accel-calibrate --dry-run --input <suite_results.json>` with an existing suite output to watch the preview.
3. For harness telemetry work, execute `python benchmarks/.harness/run_suite.py --suite benchmarks/.harness/suite.yaml --output benchmarks/results/suite_results.json` (expect long runs) and inspect `results/` for the aggregated JSON.

**Key Files to Watch**
- `benchmarks/.harness/run_suite.py` — new calibration summaries recorded in `auto_offload_calibration` sections.
- `crates/runmat-accelerate/src/native_auto.rs` — calibration ingestion logic & where future policy changes live.
- `runmat/src/main.rs` — CLI wiring for `accel-calibrate`; update this as you expose new runtime insights.

---

### 6. Coordination Notes

- **Team A**: ImageNormalize kernel already wired; upcoming vec4/logical-transpose counters should surface through provider telemetry. Stay aligned on counter naming so harness changes consume the right keys.
- **Team D**: Calibration summary structure is ready for ingestion. Loop them in once you add fused-path counters so dashboards can reflect the new data.
- **Team C**: No new parser/VM changes required; they’re aware of deterministic seeding usage in the fusion tests.

---

### 7. Open Questions / Heads-Up

- Calibration JSON schema is new—if you change field names in `_summarize_runmat_telemetry`, update both the runtime parser and CLI formatting.
- Cached thresholds are stored under `~/.cache/runmat/auto_offload/`. Keep an eye on version bumps (`CALIBRATION_VERSION`) to avoid stale loads.
- Consider capturing GPU-side telemetry (vec4, logical transpose) as part of the same suite summary to minimize schema churn.

Good luck! Ping if anything is unclear—we tried to leave the house tidy, but there’s plenty of fun left. 🙌