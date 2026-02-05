# RunMat CLI

RunMat ships with a modern, ergonomic command-line interface designed for both
everyday users and power users. It goes far beyond what MATLAB's `matlab` CLI
offers: first-class subcommands, rich help, environment-variable integration,
snapshot management, a built-in benchmarker, a Jupyter kernel, and deep control
over the JIT compiler and garbage collector.

**Try RunMat without installing:** the [browser sandbox](https://runmat.com/sandbox) runs in your browser with no CLI. For installation and CLI vs browser paths, see [Getting Started](/docs/getting-started).

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

Global options apply to all commands. Commands offer task-oriented workflows.

## Global options

- `--version`, `-V`: print RunMat version and exit (use `runmat version --detailed` for component breakdown).
- `--debug` (env: `RUNMAT_DEBUG`): enable debug logging.
- `--log-level <error|warn|info|debug|trace>` (env: `RUNMAT_LOG_LEVEL`, default: `warn`).
- `--timeout <secs>` (env: `RUNMAT_TIMEOUT`, default: `300`): execution timeout.
- `--callstack-limit <n>` (env: `RUNMAT_CALLSTACK_LIMIT`, default: `200`): call stack frames retained.
- `--error-namespace <name>` (env: `RUNMAT_ERROR_NAMESPACE`, default: `RunMat`): error identifier prefix.
- `--config <path>` (env: `RUNMAT_CONFIG`): load configuration file.
- `--no-jit` (env: `RUNMAT_JIT_DISABLE`): disable JIT (interpreter only).
- `--jit-threshold <n>` (env: `RUNMAT_JIT_THRESHOLD`, default: `10`): hotspot threshold.
- `--jit-opt-level <none|size|speed|aggressive>` (env: `RUNMAT_JIT_OPT_LEVEL`, default: `speed`).
- `--gc-preset <low-latency|high-throughput|low-memory|debug>` (env: `RUNMAT_GC_PRESET`).
- `--gc-young-size <MB>` (env: `RUNMAT_GC_YOUNG_SIZE`): young generation size.
- `--gc-threads <n>` (env: `RUNMAT_GC_THREADS`): max GC threads.
- `--gc-stats` (env: `RUNMAT_GC_STATS`): collect and display GC statistics.
- `--verbose`: verbose REPL/execution output.
- `--snapshot <path>` (env: `RUNMAT_SNAPSHOT_PATH`): preload snapshot.
- `--plot-mode <auto|gui|headless|jupyter>` (env: `RUNMAT_PLOT_MODE`).
- `--plot-headless` (env: `RUNMAT_PLOT_HEADLESS`): force headless.
- `--plot-backend <auto|wgpu|static|web>` (env: `RUNMAT_PLOT_BACKEND`).
- `--generate-config`: print a sample config to stdout.
- `--install-kernel`: install the Jupyter kernel.

Environment booleans accept: `1/0`, `true/false`, `yes/no`, `on/off`,
`enable/disable`.

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

Env vars: `RUNMAT_KERNEL_IP`, `RUNMAT_KERNEL_KEY`, and optional port vars
`RUNMAT_SHELL_PORT`, `RUNMAT_IOPUB_PORT`, `RUNMAT_STDIN_PORT`,
`RUNMAT_CONTROL_PORT`, `RUNMAT_HB_PORT`.

- `kernel-connection <connection.json>`: start with an existing connection file.

## Commands

### pkg (coming soon)
Package manager commands. These are not yet released; the binary will acknowledge the command and exit
with status 0 once the initial scaffolding lands.

```sh
runmat pkg add <name>[@version]
runmat pkg remove <name>
runmat pkg install
runmat pkg update
runmat pkg publish
```

All pkg subcommands currently print: "RunMat package manager is coming soon. Track progress in the repo."

### fs
Remote filesystem commands. These require `runmat login` and a configured project.

```sh
runmat fs ls /data
runmat fs read /data/example.mat --output example.mat
runmat fs write /data/example.mat ./example.mat
runmat fs mkdir /data/new --recursive
runmat fs rm /data/example.mat

# Manifest workflows
runmat fs manifest-history /data/dataset
runmat fs manifest-restore <version-id>
runmat fs manifest-update /data/dataset --base-version <version-id> --manifest ./manifest.json

# Version history
runmat fs history /data/example.mat
runmat fs restore <version-id>
runmat fs history-delete <version-id>

# Snapshots
runmat fs snapshot-list
runmat fs snapshot-create --message "baseline" --tag baseline
runmat fs snapshot-restore <snapshot-id>
runmat fs snapshot-delete <snapshot-id>
runmat fs snapshot-tag-list
runmat fs snapshot-tag-set <snapshot-id> <tag>
runmat fs snapshot-tag-delete <tag>

# Git sync
runmat fs git-clone ./project-repo
runmat fs git-pull
runmat fs git-push
```

### remote
Run scripts against the remote filesystem.

```sh
runmat remote run /scripts/job.m --project <project-id> --server https://api.runmat.example
```

If `--server` is omitted, the CLI defaults to `https://api.runmat.com`.

### project select
Set the default project for remote filesystem commands.

```sh
runmat project select <project-id>
```

### project retention
Manage per-project version retention.

```sh
runmat project retention get [--project <project-id>]
runmat project retention set <max-versions> [--project <project-id>]
```

`max-versions=0` disables pruning (unlimited history).

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

### accel-info
Show acceleration provider information: device name/backend, fused pipeline cache
hits/misses, last warmup duration, and reduction defaults (two-pass threshold and workgroup size).

```sh
runmat accel-info [--json] [--reset]
```

- `--json`: output provider information and telemetry as JSON.
- `--reset`: reset provider telemetry counters after printing.

Notes:
- When built without the `wgpu` feature, the command reports that no GPU provider
  is available.
- Reduction defaults can be overridden at runtime via environment variables
  (see below).
- Warmup duration reflects the provider's most recent warmup pass (including on-disk cache precompile),
  when available.

### accel-calibrate
Apply auto-offload calibration from benchmark-suite telemetry results. Available when RunMat is built with the `wgpu` feature.

```sh
runmat accel-calibrate <suite-results.json> [--dry-run] [--json]
```

- `<suite-results.json>`: path to suite results JSON produced by the benchmark harness.
- `--dry-run`: preview updates without persisting the calibration cache.
- `--json`: emit calibration outcome as JSON.

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
  - `RUNMAT_DEBUG` (bool)
  - `RUNMAT_LOG_LEVEL` = `error|warn|info|debug|trace`
  - `RUNMAT_TIMEOUT` (seconds)
  - `RUNMAT_CONFIG` (config file path)
  - `RUNMAT_SNAPSHOT_PATH` (snapshot to preload)

- JIT:
  - `RUNMAT_JIT_ENABLE` (bool, default true)
  - `RUNMAT_JIT_DISABLE` (bool, if true overrides enable)
  - `RUNMAT_JIT_THRESHOLD` (integer)
  - `RUNMAT_JIT_OPT_LEVEL` = `none|size|speed|aggressive`

- GC:
  - `RUNMAT_GC_PRESET` = `low-latency|high-throughput|low-memory|debug`
  - `RUNMAT_GC_YOUNG_SIZE` (MB)
  - `RUNMAT_GC_THREADS` (integer)
  - `RUNMAT_GC_STATS` (bool)

- Plotting:
  - `RUNMAT_PLOT_MODE` = `auto|gui|headless|jupyter`
  - `RUNMAT_PLOT_HEADLESS` (bool)
  - `RUNMAT_PLOT_BACKEND` = `auto|wgpu|static|web`

- Kernel:
  - `RUNMAT_KERNEL_IP`, `RUNMAT_KERNEL_KEY`
  - Optional ports: `RUNMAT_SHELL_PORT`, `RUNMAT_IOPUB_PORT`, `RUNMAT_STDIN_PORT`, `RUNMAT_CONTROL_PORT`, `RUNMAT_HB_PORT`

### Acceleration provider (RunMat Accelerate)

These control GPU workgroup sizing, reductions, and provider debugging:

- `RUNMAT_WG` (u32)
  - Global compute workgroup size used in WGSL at module creation.
    Applies to elementwise kernels and fused kernels (including fused reductions).
    Default: `512`.
- `RUNMAT_MATMUL_TILE` (u32)
  - Square tile size for matmul kernels. Default: `16`.
- `RUNMAT_REDUCTION_WG` (u32)
  - Default workgroup size for provider-managed reduction kernels when a call site
    opts into provider defaults (e.g., passes `0`). Default: `512`.
- `RUNMAT_TWO_PASS_THRESHOLD` (usize)
  - Threshold for reduction length per slice above which the provider uses a two-pass kernel.
    Default: `1024`.
- `RUNMAT_DEBUG_PIPELINE_ONLY` (bool)
  - When set, the provider stops after pipeline compilation and skips buffer
    creation/dispatch. Useful for triaging driver pipeline creation behavior.
- `RUNMAT_PIPELINE_CACHE_DIR` (path)
  - Overrides the on-disk pipeline cache directory. Defaults to the OS cache
    directory (e.g., `$XDG_CACHE_HOME/runmat/pipelines` or platform equivalent),
    falling back to `target/tmp/wgpu-pipeline-cache-<device>` when not set.

The provider clamps `RUNMAT_WG`, `RUNMAT_MATMUL_TILE`, and
`RUNMAT_REDUCTION_WG` to the adapter's compute limits
(`max_compute_workgroup_size_*`, `max_compute_invocations_per_workgroup`)
so DX12/Metal/Vulkan backends never see invalid workgroup sizes.

## Precedence model

CLI flags > environment variables > configuration files > built-in defaults.
See `/docs/configuration` for file formats and discovery.

## Operating environments

RunMat's CLI is designed to work identically in interactive shells and
non-interactive environments:

- CI/CD: set headless plotting and deterministic JIT options.
  - Env hints: `CI`, `GITHUB_ACTIONS`, `HEADLESS`, `NO_GUI` — presence is enough (any value).
  - To force via config var: set `RUNMAT_PLOT_MODE=headless` (must be exactly `headless`).
  - Example (GitHub Actions):
    ```sh
    RUNMAT_PLOT_MODE=headless RUNMAT_JIT_DISABLE=1 runmat run tests/current_feature_test.m
    ```
- Docker/containers: no display required; headless by default when `NO_GUI` or `HEADLESS` is set.
  ```Dockerfile
  FROM debian:stable-slim
  # install runmat binary (copy or package) and dependencies
  ENV NO_GUI=1 RUNMAT_PLOT_MODE=headless
  CMD ["runmat", "info"]
  ```
- Headless servers/HPC: disable GUI and tune GC/JIT:
  ```sh
  RUNMAT_PLOT_MODE=headless RUNMAT_GC_PRESET=high-throughput runmat benchmark perf.m --iterations 10 --jit
  ```

RunMat also auto-detects headless contexts and falls back to safe defaults.

## Examples

### Run a simple script with JIT disabled
```sh
RUNMAT_JIT_DISABLE=1 runmat my_script.m
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
