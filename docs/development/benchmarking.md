# Benchmarking

RunMat has two benchmarking utilities in this repo. The CLI `benchmark` command measures one MATLAB script through the normal session engine. The `benchmarks/` harness compares RunMat against other implementations across representative workloads and records telemetry for performance analysis.

## CLI Benchmark Command

Use `runmat benchmark` when validating a single script or a focused runtime change:

```bash
runmat benchmark script.m --iterations 5 --jit
```

The command resolves the script, creates a session, warms up three times, runs the requested iteration count, and prints:

- total iterations
- JIT execution count
- interpreter execution count
- total time
- average time
- throughput

Use `--jit` when measuring JIT behavior. Without it, the benchmark runs through the default session configuration.

## Cross-Language Benchmark Suite

The `benchmarks/` directory contains reproducible workload comparisons. Each case has a RunMat script and comparable implementations in Python, PyTorch, Julia, or Octave where applicable.

So far, the following cases are implemented, with more to come:

| Case | Purpose |
| --- | --- |
| `elementwise-math` | Long elementwise chains that stress fusion and memory bandwidth. |
| `4k-image-processing` | Image-sized array transforms and parity checks. |
| `monte-carlo-analysis` | Monte Carlo GBM-style workloads and stochastic update chains. |

The harness prefers a `runmat` binary on `PATH`. If one is not available, it falls back to `cargo run -q -p runmat --release --`, which is useful but slower because Cargo participates in the run.

Build a release binary before collecting numbers:

```bash
cargo build -p runmat --release --features wgpu
```

Run one case:

```bash
cd benchmarks
python3 ./.harness/run_bench.py \
  --case 4k-image-processing \
  --iterations 3 \
  --output ../results/4k_image_processing.json
```

Run the suite and generate plots:

```bash
cd benchmarks
python3 ./.harness/run_suite.py \
  --suite ./.harness/suite.json \
  --output ../results/suite_results.json

python3 ./.harness/plot_suite.py \
  --input ../results/suite_results.json \
  --output_dir ../results
```

Suite output is JSON. Plot output includes scaling, speedup, and telemetry charts.

## Parity And Metrics

Benchmark cases print `RESULT_ok` markers with case-specific metrics such as MSE or price. The suite config defines the expected regexes and parity checks, then records pass/fail details in the result JSON, allowing the harness to compare numerical results and program outputs between different underlying implementations.

`run_bench.py` records implementation name, language, iteration count, median time, all timings, stdout and stderr tails, optional case metrics, device data, and RunMat provider telemetry. Treat checked-in result files as historical artifacts; regenerate results on the machine and GPU being evaluated.

## GPU Telemetry

The benchmark harness can capture RunMat acceleration telemetry by setting `RUNMAT_TELEMETRY_OUT` and `RUNMAT_TELEMETRY_RESET` for RunMat runs. The CLI also exposes direct acceleration inspection:

```bash
runmat accel-info
runmat accel-info --json --reset
runmat accel-calibrate benchmarks/results/suite_results.json --dry-run --json
runmat accel-calibrate --refresh --dry-run --json
runmat accel-calibrate --refresh --json
```

Use telemetry when deciding whether a benchmark is measuring the intended execution path. For GPU work, verify provider selection, offload decisions, residency behavior, and fallback counts before interpreting raw timings.

Suite residual CPU-time/category allocations are diagnostic estimates, not isolated CPU measurements. The harness marks them with schema 2 and `diagnostic-residual-v1`; importing them with `--dry-run` reports unchanged coefficients and `residual-diagnostic` provenance. They cannot be committed as calibration. `--auto-calibrate` now invokes `accel-calibrate --refresh`, which reuses the runtime's existing original CPU templates, provider comparisons, and cleanup owner. `--dry-run` measures without persisting; a committed refresh writes the provider/precision-specific v2 cache. Automatic sample counts are not fabricated: the refresh outcome has `runs: null` because that owner does not export a sample census. Input files and `--refresh` are mutually exclusive.

Qualified original-template imports require schema 2, the closed `original-scan-mac-v1` unit policy, exact provider identity, and `F32`/`F64` precision. Historical scan/MAC denominators are unchanged. Output publication/conversion is separately identified cold-prior work, not a relabeled template measurement. Legacy category profile fitting, `RUNMAT_ACCEL_PROFILE`, and auto-offload `profile_path` no longer participate in decisions or calibration. `wgpu_profile` remains a diagnostic benchmark; matched whole-call observations belong to the canonical PlacementSession owner.

GPU submission time measures how long the host takes to enqueue work; it does not measure completed device execution. Fusion placement traces report successful submission as a queue event, without a completion event or a successful execution-cost sample. Measure device completion and any required transfers or synchronization when comparing CPU and GPU latency. A CPU-selected fusion region records completion only after its bytecode reaches the region's normal exit; errors, cancellation, and abandoned execution do not contribute successful timing samples.

On Metal, RunMat uses compute-pass timestamp queries but does not enable encoder-wide timestamp writes, which can prevent submissions from completing. The full-command device interval is therefore unavailable; it is not replaced by the sum of compute-pass times or a host synchronization time. Compare completed end-to-end latency when transfer and synchronization costs matter.

## Performance Smoke Tests

Some performance checks live as Rust tests. They are not a replacement for benchmark runs, but they catch severe regressions in JIT, residency, and GPU dispatch behavior.

```bash
cargo test -p runmat-jit
cargo test -p runmat-core native_tiering
cargo test -p runmat-runtime-integration-tests --test bench_residency_smoke
cargo test -p runmat-accelerate --features wgpu --test matmul_residency
cargo test -p runmat-vm --test fusion_gpu
```

The JIT and Core tiering tests check bounded warmup, publication and cache reuse, exact-profile specialization, loop feedback and OSR, transactional interactive execution, and steady-state compilation counts. Residency tests check that GPU values remain on device across representative operation chains.

## Direct-call Overhead

The direct-call microbenchmark compares scalar `strcmp` and scalar/array `numel` calls with and without a placement session:

```bash
cargo bench -p runmat-accelerate --bench direct_call_observation --no-default-features -- 10000 31
```

The arguments select repeated-call iterations and sample count. CSV output includes the workload, placement mode, first-call or repeated-call phase, sample number, iteration count, and nanoseconds per call. Inputs and contexts are constructed outside timing; each measured call includes dispatch, execution, result creation, and result release. Results are checked afterward.

First-call samples use fresh sessions, not fresh processes. Repeated-call samples prime their session before timing and continue updating its feedback. The placement/no-placement contrast measures the whole placement-enabled call path, not allocation cost alone. To measure a change's overhead, run the same benchmark on both source revisions with matching compiler, features, settings, and machine conditions. These host-only measurements do not establish CPU/GPU speedups or end-to-end script performance.

## WGPU Profiling

The acceleration crate includes a WGPU profiling binary for backend-level measurements:

```bash
cargo run -p runmat-accelerate --features wgpu --bin wgpu_profile -- --output wgpu_profile.json
cargo run -p runmat-accelerate --features wgpu --bin wgpu_profile -- --quick --reduce-sweep --output wgpu_profile.json
cargo run -p runmat-accelerate --features wgpu --bin wgpu_profile -- --kernel-probe
```

Use this tool for WGPU backend questions that need lower-level measurements than the CLI or benchmark harness provides.
