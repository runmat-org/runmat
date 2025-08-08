### RustMat Performance Benchmarks

This suite compares RustMat and GNU Octave across representative MATLAB-style workloads. It builds RustMat (if needed), runs each script in Octave, RustMat interpreter, and RustMat JIT, and writes a YAML report with system info and speedups.

### What’s included

- **Startup Time** (`startup_time.m`): end‑to‑end cold script execution.
- **Matrix Operations** (`matrix_operations.m`): add, matmul, transpose, scalar‑mul over sizes `[100, 300, 500]`.
- **Math Functions** (`math_functions.m`): `sin, cos, exp, log, sqrt, sum, mean, std` over `[50k, 200k, 500k]`.
- **Control Flow** (`control_flow.m`): simple loop, nested loops, conditionals, and function calls over `n = [1k, 5k, 10k]`.

Per‑size timings are printed by the scripts via `fprintf`, while the runner measures total wall‑clock time per script (warmup + repeated runs) for the YAML.

### Prerequisites

- Rust toolchain and Cargo
- `bc` (for basic math in the runner)
- Optional: `yq` (pretty‑printing YAML locally)
- Optional: GNU Octave in `PATH` (for Octave comparisons)
  - macOS: `brew install octave`
  - Ubuntu/Debian: `sudo apt-get install octave`

### Quick start

```bash
cd benchmarks
./run_benchmarks.sh
```

What the runner does:
- Builds RustMat (release) if missing
- Gathers OS/CPU/memory and tool versions
- For each script: 1 warmup + 3 timed runs for Octave, interpreter, and JIT
- Computes avg/min/max and speedups; writes `results/benchmark_<timestamp>.yaml`

### Manual use

```bash
# Octave
octave --no-gui matrix_operations.m

# RustMat interpreter only
../target/release/rustmat --no-jit matrix_operations.m

# RustMat with JIT
../target/release/rustmat matrix_operations.m
```

### Output format (YAML)

```yaml
metadata:
  timestamp: "YYYYMMDD_HHMMSS"
  date: "ISO-8601"
system:
  os: Darwin|Linux
  architecture: arm64|x86_64
  cpu:
    model: "Apple M2 Max"
    cores: 12
  memory_gb: 32.0
software:
  octave:
    version: "9.4.0"        # if available
  rustmat:
    version: "rustmat 0.0.1"
    build_features: ["blas-lapack"]
benchmark_config:
  warmup_runs: 1
  timing_runs: 3
  scripts: [startup_time.m, matrix_operations.m, math_functions.m, control_flow.m]
results:
  matrix_operations:
    octave: { avg_time: 0.822, min_time: 0.819, max_time: 0.826 }
    rustmat_interpreter: { avg_time: 0.005, min_time: 0.005, max_time: 0.005, speedup_vs_octave: "164.40x" }
    rustmat_jit: { avg_time: 0.005, min_time: 0.005, max_time: 0.005, speedup_vs_octave: "164.40x", speedup_vs_interpreter: "1.00x" }
```

Note: The YAML aggregates total script time. Per‑size timings in the script output are for human inspection and are not parsed into YAML.

### Formatting and logging

- `fprintf` in RustMat supports `%d`, `%f`, and `%.Nf` plus `\n` (newline). Prefer `%.6f` for timings.
- Scripts use `tic`/`toc` for intra‑script measurements; the runner uses external wall‑clock time.

### Customize sizes

Edit within the scripts:
- `matrix_operations.m`: `sizes = [100, 300, 500];`
- `math_functions.m`: `sizes = [50000, 200000, 500000];`
- `control_flow.m`: `iterations = [1000, 5000, 10000];`

### Adding a new benchmark

- Create a `.m` file that:
  - prints a header with `fprintf`,
  - measures its sections with `tic`/`toc`,
  - prints numeric results using `%` formats (e.g., `%.6f`).
- Add the script (without extension) to the `benchmarks` array in `run_benchmarks.sh`.
- Re‑run the suite.

### Troubleshooting

- **Octave times are N/A**: Ensure Octave is installed and on `PATH`.
- **Permissions**: `chmod +x run_benchmarks.sh`.
- **Headless environments**: All benchmarks are non‑GUI; Octave runs with `--no-gui`.

### Reproducibility notes

- System metadata and tool versions are embedded in each YAML.
- BLAS/LAPACK features are recorded in `software.rustmat.build_features`.
- For cross‑machine comparisons, run on AC power with minimal background load.