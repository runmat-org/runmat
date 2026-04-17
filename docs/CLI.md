# RunMat CLI

RunMat ships with a task-oriented command-line interface for running scripts,
working interactively, operating Jupyter kernels, and managing remote
project-backed storage. Use this page as a workflow guide first, then fall back
to `runmat --help` and `runmat <command> --help` for the full generated
reference.

**Try RunMat without installing:** the [browser sandbox](https://runmat.com/sandbox)
runs in your browser with no CLI. For installation and broader product setup,
see [Getting Started](/docs/getting-started).

## Installation

```bash
# Quick install

## Linux/macOS
curl -fsSL https://runmat.com/install.sh | sh

## Windows PowerShell
iwr https://runmat.com/install.ps1 | iex

# Alternative installation methods

## Homebrew (macOS/Linux)
brew install runmat-org/tap/runmat

## Cargo
cargo install runmat --features gui

## Build from source
git clone https://github.com/runmat-org/runmat.git
cd runmat && cargo build --release --features gui
```

## Quick start

```sh
# Start the REPL
runmat

# Run a local script
runmat model.m
```

## Usage

```text
runmat [GLOBAL OPTIONS] [COMMAND] [ARGS]
runmat [GLOBAL OPTIONS] <script.m>
```

RunMat supports both:
- direct script execution: `runmat my_script.m`
- explicit subcommands: `runmat repl`, `runmat benchmark ...`, `runmat project fs ls ...`

## Core workflows

### Run a script

```sh
runmat run <file.m> [-- arg1 arg2 ...]

# shorthand
runmat <file.m>
```

Useful variants:

```sh
# Emit bytecode to stdout
runmat --emit-bytecode model.m

# Emit bytecode to a file
runmat --emit-bytecode bytecode.txt model.m

# Capture artifacts and exported figures
runmat model.m \
  --artifacts-dir .runmat-artifacts \
  --capture-figures auto \
  --figure-size 1280x720
```

When `--artifacts-dir` or `--artifacts-manifest` is set, RunMat writes a JSON
manifest describing execution metadata, stream sizes, touched figure handles,
and exported figure paths.

### Interactive REPL

```sh
runmat
runmat repl
runmat repl --verbose
```

Built-in REPL commands:
- `.info`: detailed system information
- `.stats`: execution statistics
- `.gc`: garbage collector summary
- `.gc-info`: garbage collector summary with header
- `.gc-collect`: force garbage collection
- `.reset-stats`: reset execution statistics
- `help`: show REPL help
- `exit`, `quit`: leave the REPL

### Jupyter kernel

```sh
# Install the kernel spec
runmat --install-kernel

# Start a kernel directly
runmat kernel

# Start from an existing connection file
runmat kernel-connection connection.json
```

Advanced kernel flags exist for IP, ports, transport, signature scheme, and
connection-file output. Use `runmat kernel --help` when you need to wire RunMat
into an existing Jupyter environment manually.

### Diagnostics

```sh
runmat info
runmat version
runmat version --detailed
```

- `info` prints runtime configuration, selected environment variables, GC
  status, and pointers to built-in help.
- `version --detailed` prints build details useful in bug reports and support
  threads.

## Remote and project workflows

Remote commands are project-scoped. A typical flow is:

1. authenticate with `runmat login`
2. inspect orgs and projects
3. select a default project
4. use `project fs ...` or `remote run ...`

### Authenticate

```sh
runmat login --server https://api.runmat.com
```

You can also supply:
- `--api-key <token>` for non-interactive login
- `--email <address>` for interactive login
- `--credential-store <auto|secure|file|memory>`
- `--org <org-id>` and `--project <project-id>` to seed defaults

### Inspect and select projects

```sh
runmat org list

runmat project list
runmat project list --org <org-id>
runmat project create my-project
runmat project members list
runmat project select <project-id>
runmat project retention get
runmat project retention set 20
```

`project retention set 0` means unlimited history.

### Remote files

The canonical namespace is `project fs ...`:

```sh
runmat project fs ls /data
runmat project fs read /data/example.mat --output example.mat
runmat project fs write /data/example.mat ./example.mat
runmat project fs mkdir /data/new --recursive
runmat project fs rm /data/example.mat
```

The shorter top-level `fs ...` form still works as a shorthand:

```sh
runmat fs ls /data
```

The filesystem surface also includes:
- file history: `history`, `restore`, `history-delete`
- manifest history: `manifest-history`, `manifest-restore`, `manifest-update`
- project snapshots: `snapshot-list`, `snapshot-create`, `snapshot-restore`,
  `snapshot-delete`, `snapshot-tag-*`
- git sync: `git-clone`, `git-pull`, `git-push`

Use `runmat project fs --help` or `runmat fs --help` for the full subcommand
tree.

### Run a remote script

```sh
runmat remote run /scripts/job.m
runmat remote run /scripts/job.m --project <project-id>
runmat remote run /scripts/job.m --server https://api.runmat.com
```

`remote run` loads the script from the remote filesystem, then executes it
using the current RunMat CLI configuration. This is different from `runmat
local_script.m`, which reads the source from local disk.

## Tooling and diagnostics

### Configuration

```sh
runmat config show
runmat config generate -o .runmat.yaml
runmat config validate .runmat.yaml
runmat config paths
```

### Snapshots

```sh
runmat snapshot create -o stdlib.snapshot --compression zstd -O speed
runmat snapshot info stdlib.snapshot
runmat snapshot presets
runmat snapshot validate stdlib.snapshot
```

### Benchmarking

```sh
runmat benchmark <file.m> [--iterations N] [--jit]
```

`--iterations` defaults to `10`.

### Garbage collection

```sh
runmat gc stats
runmat gc minor
runmat gc major
runmat gc config
runmat gc stress --allocations 10000
```

### Acceleration diagnostics

```sh
runmat accel-info [--json] [--reset]
runmat accel-calibrate <suite-results.json> [--dry-run] [--json]
```

- `accel-info` prints provider and telemetry details.
- `accel-calibrate` is available when RunMat is built with the `wgpu` feature.
- For feature-specific or GPU-provider-specific behavior, prefer the command
  help and acceleration-focused docs over this page.

## Global flags

Global flags apply to both direct script execution and subcommands.

### Logging and control

- `--version`, `-V`: print the version and exit
- `--debug`: enable debug logging
- `--log-level <error|warn|info|debug|trace>`: set log verbosity
- `--config <path>`: load a configuration file
- `--generate-config`: print a sample config to stdout

### Execution and diagnostics

- `--timeout <secs>`: execution timeout, default `300`
- `--callstack-limit <n>`: retained call-stack frames, default `200`
- `--error-namespace <name>`: error identifier prefix
- `--verbose`: verbose REPL / execution output
- `--snapshot <path>`: preload a standard-library snapshot
- `--emit-bytecode [PATH]`: emit bytecode instead of executing a script

### JIT and GC

- `--no-jit`: disable JIT compilation
- `--jit-threshold <n>`: JIT trigger threshold, default `10`
- `--jit-opt-level <none|size|speed|aggressive>`: JIT optimization level
- `--gc-preset <low-latency|high-throughput|low-memory|debug>`
- `--gc-young-size <MB>`
- `--gc-threads <n>`
- `--gc-stats`

### Plotting and artifacts

- `--plot-mode <auto|gui|headless|jupyter>`
- `--plot-headless`
- `--plot-backend <auto|wgpu|static|web>`
- `--plot-scatter-target <n>`
- `--plot-surface-vertex-budget <n>`
- `--artifacts-dir <path>`
- `--artifacts-manifest <path>`
- `--capture-figures <off|auto|on>`
- `--figure-size <WIDTHxHEIGHT>`
- `--max-figures <n>`

### Integrations

- `--install-kernel`: install the RunMat Jupyter kernel

## Environment variables

Most execution-related globals also have environment variable forms. Boolean
values accept `1/0`, `true/false`, `yes/no`, `on/off`, and `enable/disable`.

### General

- `RUNMAT_DEBUG`
- `RUNMAT_LOG_LEVEL`
- `RUNMAT_TIMEOUT`
- `RUNMAT_CALLSTACK_LIMIT`
- `RUNMAT_ERROR_NAMESPACE`
- `RUNMAT_CONFIG`
- `RUNMAT_SNAPSHOT_PATH`

### JIT and GC

- `RUNMAT_JIT_ENABLE`
- `RUNMAT_JIT_DISABLE`
- `RUNMAT_JIT_THRESHOLD`
- `RUNMAT_JIT_OPT_LEVEL`
- `RUNMAT_GC_PRESET`
- `RUNMAT_GC_YOUNG_SIZE`
- `RUNMAT_GC_THREADS`
- `RUNMAT_GC_STATS`

### Plotting and artifacts

- `RUNMAT_PLOT_MODE`
- `RUNMAT_PLOT_HEADLESS`
- `RUNMAT_PLOT_BACKEND`
- `RUNMAT_ARTIFACTS_DIR`
- `RUNMAT_ARTIFACTS_MANIFEST`
- `RUNMAT_CAPTURE_FIGURES`
- `RUNMAT_FIGURE_SIZE`
- `RUNMAT_MAX_FIGURES`

### Kernel

- `RUNMAT_KERNEL_IP`
- `RUNMAT_KERNEL_KEY`
- `RUNMAT_SHELL_PORT`
- `RUNMAT_IOPUB_PORT`
- `RUNMAT_STDIN_PORT`
- `RUNMAT_CONTROL_PORT`
- `RUNMAT_HB_PORT`

## Precedence

CLI flags override environment variables, which override configuration files,
which override built-in defaults.

See `/docs/configuration` for configuration file discovery and formats.

## CI, headless, and containers

RunMat is intended to work in both interactive shells and non-interactive
environments.

```sh
# Force headless execution in CI
RUNMAT_PLOT_MODE=headless RUNMAT_JIT_DISABLE=1 runmat tests/current_feature_test.m

# Headless benchmark run
RUNMAT_PLOT_MODE=headless RUNMAT_GC_PRESET=high-throughput \
  runmat benchmark perf.m --iterations 10 --jit
```

```Dockerfile
FROM debian:stable-slim
# install runmat binary and runtime dependencies
ENV NO_GUI=1 RUNMAT_PLOT_MODE=headless
CMD ["runmat", "info"]
```

## Exit codes

- `0`: success
- `1`: command execution, runtime, validation, or file/config errors
- `2`: invalid CLI usage (for example, unknown flags or malformed arguments)

## Not yet available

`runmat pkg ...` exists as a placeholder command family, but the package manager
is not shipped yet. The current subcommands print a "coming soon" message and
exit successfully.
