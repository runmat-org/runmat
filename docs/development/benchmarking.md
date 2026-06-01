---
title: "Benchmarking"
category: "Development"
section: "14.4"
last_updated: "May 28, 2026"
---

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
```

Use telemetry when deciding whether a benchmark is measuring the intended execution path. For GPU work, verify provider selection, offload decisions, residency behavior, and fallback counts before interpreting raw timings.

## Performance Smoke Tests

Some performance checks live as Rust tests. They are not a replacement for benchmark runs, but they catch severe regressions in JIT, residency, and GPU dispatch behavior.

```bash
cargo test -p runmat-turbine --test performance
cargo test -p runmat-runtime-integration-tests --test bench_residency_smoke
cargo test -p runmat-accelerate --features wgpu --test matmul_residency
cargo test -p runmat-vm --test fusion_gpu
```

The Turbine performance tests check basic compile time, execution time, cache behavior, repeated execution, and scalability expectations. Residency tests check that GPU values remain on device across representative operation chains.

## WGPU Profiling

The acceleration crate includes a WGPU profiling binary for backend-level measurements:

```bash
cargo run -p runmat-accelerate --features wgpu --bin wgpu_profile -- --output wgpu_profile.json
cargo run -p runmat-accelerate --features wgpu --bin wgpu_profile -- --quick --reduce-sweep --output wgpu_profile.json
cargo run -p runmat-accelerate --features wgpu --bin wgpu_profile -- --kernel-probe
```

Use this tool for WGPU backend questions that need lower-level measurements than the CLI or benchmark harness provides.
