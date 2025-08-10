# RunMat CLI

RunMat ships with a modern, ergonomic command-line interface designed for both
everyday users and power users. It goes far beyond what MATLAB's `matlab` CLI
offers: first-class subcommands, rich help, environment-variable integration,
snapshot management, a built-in benchmarker, a Jupyter kernel, and deep control
over the JIT compiler and garbage collector.

Read this end-to-end once, then use it as a reference.

## Quick start

```sh
# Start interactive REPL (JIT on by default)
runmat

# Run a script
runmat my_script.m

# Pipe a script to RunMat
echo "x = 1 + 2" | runmat

# System info (current config, env, GC status)
runmat info

# Benchmark a script (5 iterations, enable JIT explicitly)
runmat benchmark my_script.m --iterations 5 --jit

# Create and inspect a snapshot
runmat snapshot create -o stdlib.snapshot
runmat snapshot info stdlib.snapshot
```

## Usage

```text
runmat [GLOBAL OPTIONS] [COMMAND] [ARGS]
runmat [GLOBAL OPTIONS] <script.m>
```

Global options apply to all commands. Commands offer task‑oriented workflows.

## Global options

- `--debug` (env: `RUSTMAT_DEBUG`): enable debug logging.
- `--log-level <error|warn|info|debug|trace>` (env: `RUSTMAT_LOG_LEVEL`, default: `info`).
- `--timeout <secs>` (env: `RUSTMAT_TIMEOUT`, default: `300`): execution timeout.
- `--config <path>` (env: `RUSTMAT_CONFIG`): load configuration file.
- `--no-jit` (env: `RUSTMAT_JIT_DISABLE`): disable JIT (interpreter only).
- `--jit-threshold <n>` (env: `RUSTMAT_JIT_THRESHOLD`, default: `10`): hotspot threshold.
- `--jit-opt-level <none|size|speed|aggressive>` (env: `RUSTMAT_JIT_OPT_LEVEL`, default: `speed`).
- `--gc-preset <low-latency|high-throughput|low-memory|debug>` (env: `RUSTMAT_GC_PRESET`).
- `--gc-young-size <MB>` (env: `RUSTMAT_GC_YOUNG_SIZE`): young generation size.
- `--gc-threads <n>` (env: `RUSTMAT_GC_THREADS`): max GC threads.
- `--gc-stats` (env: `RUSTMAT_GC_STATS`): collect and display GC statistics.
- `--verbose`: verbose REPL/execution output.
- `--snapshot <path>` (env: `RUSTMAT_SNAPSHOT_PATH`): preload snapshot.
- `--plot-mode <auto|gui|headless|jupyter>` (env: `RUSTMAT_PLOT_MODE`).
- `--plot-headless` (env: `RUSTMAT_PLOT_HEADLESS`): force headless.
- `--plot-backend <auto|wgpu|static|web>` (env: `RUSTMAT_PLOT_BACKEND`).
- `--generate-config`: print a sample config to stdout.
- `--install-kernel`: install the Jupyter kernel.

Environment booleans accept: `1/0`, `true/false`, `yes/no`, `on/off`,
`enable/disable`.

## Commands

### repl
Start interactive REPL.

```sh
runmat repl [--verbose]
```

Interactive commands inside the REPL:
- `.info`: detailed system information
- `.stats`: execution statistics (total, JIT vs interpreter, avg time)
- `.gc`: GC statistics summary

Example:
```text
runmat> x = 1 + 2
ans = 3
runmat> y = [1, 2; 3, 4]
ans = [2x2 matrix]
runmat> .stats
Execution Statistics:
  Total: 2, JIT: 0, Interpreter: 2
  Average time: 0.12ms
```

### run
Execute a MATLAB/Octave script file.

```sh
runmat run <file.m> [-- arg1 arg2 ...]

# shorthand
runmat <file.m>
```

Example `calc.m`:
```matlab
x = 10 + 5
```

```sh
runmat calc.m
```

### kernel
Start the Jupyter kernel.

```sh
runmat kernel \
  [--ip 127.0.0.1] [--key <string>] \
  [--transport tcp] [--signature-scheme hmac-sha256] \
  [--shell-port 0] [--iopub-port 0] [--stdin-port 0] [--control-port 0] [--hb-port 0] \
  [--connection-file <path>] 
```

Env vars: `RUSTMAT_KERNEL_IP`, `RUSTMAT_KERNEL_KEY`, and optional port vars
`RUSTMAT_SHELL_PORT`, `RUSTMAT_IOPUB_PORT`, `RUSTMAT_STDIN_PORT`,
`RUSTMAT_CONTROL_PORT`, `RUSTMAT_HB_PORT`.

- `kernel-connection <connection.json>`: start with an existing connection file.

### version
```sh
runmat version [--detailed]
```
With `--detailed`, prints component breakdown (lexer, parser, interpreter, JIT,
GC, runtime, kernel, plotting).

### info
Show structured system information: versions, CLI/runtime config, env, GC
status, and available commands.

```sh
runmat info
```

### gc
Garbage collection utilities.

```sh
runmat gc stats
runmat gc minor
runmat gc major
runmat gc config
runmat gc stress --allocations 10000
```

Example:
```text
$ runmat gc major
Major GC collected 1234 objects in 8.2ms
```

### benchmark
Built-in benchmark driver.

```sh
runmat benchmark <file.m> [--iterations N] [--jit]
```

Example `loop.m`:
```matlab
total = 0; for i = 1:1000; total = total + i; end
```

```sh
runmat benchmark loop.m --iterations 5 --jit
```

### snapshot
Manage standard library snapshots.

```sh
runmat snapshot create -o <file> [-O <none|size|speed|aggressive>] [--compression <none|lz4|zstd>]
runmat snapshot info <file>
runmat snapshot presets
runmat snapshot validate <file>
```

Examples:
```sh
runmat snapshot create -o stdlib.snapshot --compression zstd -O speed
runmat snapshot info stdlib.snapshot
```

### plot
Interactive plotting window (GUI) or headless demo plot.

```sh
runmat plot [--mode <auto|gui|headless|jupyter>] [--width W] [--height H]
```

- `headless` mode generates a sample plot (`sample_plot.png`) using the static
  backend. GUI/Jupyter options are integrated with the plotting crate and
  environment detection; Jupyter mode will be expanded in future releases.

### config
Configuration management helpers.

```sh
runmat config show
runmat config generate [-o .runmat.yaml]
runmat config validate <file>
runmat config paths
```

`generate` writes a complete sample file to help you get started.

## Environment variables (reference)

- Logging and control:
  - `RUSTMAT_DEBUG` (bool)
  - `RUSTMAT_LOG_LEVEL` = `error|warn|info|debug|trace`
  - `RUSTMAT_TIMEOUT` (seconds)
  - `RUSTMAT_CONFIG` (config file path)
  - `RUSTMAT_SNAPSHOT_PATH` (snapshot to preload)

- JIT:
  - `RUSTMAT_JIT_ENABLE` (bool, default true)
  - `RUSTMAT_JIT_DISABLE` (bool, if true overrides enable)
  - `RUSTMAT_JIT_THRESHOLD` (integer)
  - `RUSTMAT_JIT_OPT_LEVEL` = `none|size|speed|aggressive`

- GC:
  - `RUSTMAT_GC_PRESET` = `low-latency|high-throughput|low-memory|debug`
  - `RUSTMAT_GC_YOUNG_SIZE` (MB)
  - `RUSTMAT_GC_THREADS` (integer)
  - `RUSTMAT_GC_STATS` (bool)

- Plotting:
  - `RUSTMAT_PLOT_MODE` = `auto|gui|headless|jupyter`
  - `RUSTMAT_PLOT_HEADLESS` (bool)
  - `RUSTMAT_PLOT_BACKEND` = `auto|wgpu|static|web`

- Kernel:
  - `RUSTMAT_KERNEL_IP`, `RUSTMAT_KERNEL_KEY`
  - Optional ports: `RUSTMAT_SHELL_PORT`, `RUSTMAT_IOPUB_PORT`, `RUSTMAT_STDIN_PORT`, `RUSTMAT_CONTROL_PORT`, `RUSTMAT_HB_PORT`

## Precedence model

CLI flags > environment variables > configuration files > built-in defaults.
See `docs/CONFIG.md` for file formats and discovery.

## Operating environments

RunMat's CLI is designed to work identically in interactive shells and
non-interactive environments:

- CI/CD: set headless plotting and deterministic JIT options.
  - Env hints: `CI`, `GITHUB_ACTIONS`, `HEADLESS`, `NO_GUI` — presence is enough (any value).
  - To force via config var: set `RUSTMAT_PLOT_MODE=headless` (must be exactly `headless`).
  - Example (GitHub Actions):
    ```sh
    RUSTMAT_PLOT_MODE=headless RUSTMAT_JIT_DISABLE=1 runmat run tests/current_feature_test.m
    ```
- Docker/containers: no display required; headless by default when `NO_GUI` or `HEADLESS` is set.
  ```Dockerfile
  FROM debian:stable-slim
  # install runmat binary (copy or package) and dependencies
  ENV NO_GUI=1 RUSTMAT_PLOT_MODE=headless
  CMD ["runmat", "info"]
  ```
- Headless servers/HPC: disable GUI and tune GC/JIT:
  ```sh
  RUSTMAT_PLOT_MODE=headless RUSTMAT_GC_PRESET=high-throughput runmat benchmark perf.m --iterations 10 --jit
  ```

RunMat also auto-detects headless contexts and falls back to safe defaults.

## Examples

### Run a simple script with JIT disabled
```sh
RUSTMAT_JIT_DISABLE=1 runmat my_script.m
```

### Create a minimal config and run with it
```sh
runmat config generate -o .runmat.yaml
runmat --config .runmat.yaml info
```

### GC stress test
```sh
runmat gc stress --allocations 200000
```

### Snapshot for faster startup
```sh
runmat snapshot create -o stdlib.snapshot
runmat --snapshot stdlib.snapshot repl
```

### Headless demo plot export
```sh
runmat plot --mode headless
ls sample_plot.png
```

## Why this is better than MATLAB's CLI

- Modern subcommand design (repl, run, kernel, snapshot, benchmark, gc, config) instead of a
  monolithic binary with sparse flags.
- First-class JIT and GC control with real-time stats.
- Snapshot tooling for fast startup and reproducible deployments.
- Built-in benchmarker with throughput/latency reporting.
- Clean, typed configuration with YAML/JSON/TOML and clear precedence.
- Structured, readable outputs suitable for automation and CI.
- Built for modern environments: CI/CD, containers, headless servers without X.

## Exit codes

- `0`: success
- non-zero: validation errors, file not found, runtime errors, or command failures
