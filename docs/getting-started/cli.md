---
title: "Command Line Interface"
category: "Getting Started"
section: "1.3"
last_updated: "May 29, 2026"
---

# Command Line Interface

The RunMat CLI is a fast and easy way to run `.m` files locally, open an interactive REPL, inspect runtime behavior, and work with remote project filesystems.

Install RunMat first if the `runmat` command is not already on your `PATH`. See [Installation](/docs/runtime/getting-started/install) for install options.

To check the version of RunMat, run:

```bash
runmat --version
```

## REPL

Run `runmat` with no command to open the interactive REPL.

```bash
runmat
```

You can also start it explicitly:

```bash
runmat repl
runmat repl --verbose
```

The REPL keeps one session alive, so variables remain available between prompts.

```matlab
A = magic(3)
sum(A)
```

REPL commands:

| Command | Use |
| --- | --- |
| `help` | Show REPL help. |
| `exit`, `quit` | Leave the REPL. |
| `.info` | Show runtime information. |
| `.stats` | Show execution statistics. |
| `.gc`, `.gc-info` | Show garbage collector statistics. |
| `.gc-collect` | Force a major collection. |
| `.reset-stats` | Reset execution statistics. |

The REPL also accepts piped input:

```bash
printf "1 + 1\n" | runmat repl
```

## Run

Run a local `.m` file by passing the path directly:

```bash
runmat analysis.m
```

The explicit form is:

```bash
runmat run analysis.m
```

RunMat also resolves configured project entrypoints. If a project has `runmat.toml` with an entrypoint named `main`, this works:

```bash
runmat run main
```

See [Projects](/docs/runtime/getting-started/projects) for project layout and entrypoint configuration.

For relative source paths, RunMat can infer a missing `.m` extension:

```bash
runmat run src/main
```

Execution uses the same session pipeline as other hosts: parse, lower, compile, run, emit streams, update workspace, and report structured diagnostics.

## Pass Runtime Options

Global options apply to the REPL, local scripts, and most commands.

```bash
runmat --no-jit analysis.m
runmat --jit-opt-level aggressive analysis.m
runmat --gc-preset low-latency analysis.m
runmat --plot-headless analysis.m
```

Common options:

| Option | Use |
| --- | --- |
| `--config PATH` | Load a specific `runmat.toml` or `runmat.json`. |
| `--debug` | Enable debug logging. |
| `--log-level LEVEL` | Set log verbosity. |
| `--verbose` | Print more execution detail. |
| `--snapshot PATH` | Preload a runtime snapshot. |
| `--no-jit` | Use the interpreter only. |
| `--jit-threshold N` | Set the execution count before JIT tiering. |
| `--jit-opt-level LEVEL` | Set JIT optimization policy. |
| `--gc-preset PRESET` | Select a GC tuning preset. |
| `--gc-young-size MB` | Override young generation size. |
| `--gc-threads N` | Override GC worker count. |
| `--gc-stats` | Collect GC statistics. |
| `--plot-mode MODE` | Select plotting mode (auto | gui | headless). |
| `--plot-headless` | Force headless plotting. |
| `--plot-backend BACKEND` | Select plotting backend (auto | wgpu | static | web). |

Configuration is resolved from built-in defaults, project files, environment variables, and CLI flags. CLI flags have the highest precedence. See [Configuration Reference](/docs/runtime/getting-started/config).

## Emit Bytecode

Use bytecode output when debugging the compiler pipeline or checking what a script lowers into before execution.

```bash
runmat --emit-bytecode analysis.m
```

Write the disassembly to a file:

```bash
runmat --emit-bytecode bytecode.txt analysis.m
```

When bytecode emission is enabled, the script is compiled and disassembled instead of being executed.

## Capture Artifacts

For batch jobs, CI, and notebook-style hosts, the CLI can write a run manifest and exported figure images.

```bash
runmat \
  --artifacts-dir .runmat-artifacts \
  --capture-figures auto \
  --figure-size 1280x720 \
  analysis.m
```

The manifest records execution metadata, stream sizes, touched figure handles, figure export paths, JIT usage, and any error identifier. Figure capture writes PNG files under the artifact directory when figures are touched or when capture is forced on.

Artifact options:

| Option | Use |
| --- | --- |
| `--artifacts-dir PATH` | Directory for run artifacts. |
| `--artifacts-manifest PATH` | Exact JSON manifest path. |
| `--capture-figures MODE` | Figure export policy (off | auto | on). |
| `--figure-size WIDTHxHEIGHT` | Figure export dimensions. |
| `--max-figures N` | Maximum number of touched figures to export. |

## Inspect Runtime

Use these commands when filing issues, tuning performance, or checking what runtime configuration is active.

```bash
runmat info
runmat version --detailed
runmat gc stats
runmat accel-info
```

| Command | Use |
| --- | --- |
| `info` | Print version, runtime configuration, environment, and GC status. |
| `version --detailed` | Print build details useful for support and bug reports. |
| `gc stats` | Print current GC counters. |
| `gc minor`, `gc major` | Force a minor or major collection. |
| `gc config` | Print current GC configuration. |
| `accel-info` | Print acceleration provider and telemetry details. |
| `accel-info --json` | Emit acceleration details as JSON. |

## Configuration

Generate a starter config:

```bash
runmat config generate -o runmat.toml
```

Inspect resolved configuration:

```bash
runmat config show --format toml
runmat config show --format json
```

Validate and locate config files:

```bash
runmat config validate runmat.toml
runmat config paths
```

`config generate` writes both project and runtime sections, so the generated file can be used as a starting point for named entrypoints and runtime tuning.

## Benchmark

Benchmark a script or named entrypoint with repeated execution in one session.

```bash
runmat benchmark analysis.m --iterations 25
runmat benchmark main --iterations 25 --jit
```

The benchmark command performs warmup runs, then reports total iterations, JIT executions, interpreter executions, total time, average time, and throughput.

## Snapshots

Snapshots preload runtime assets so startup does less work.

```bash
runmat snapshot create -o stdlib.snapshot --compression zstd -O speed
runmat snapshot info stdlib.snapshot
runmat snapshot validate stdlib.snapshot
runmat snapshot presets
```

Load a snapshot for a script or REPL:

```bash
runmat --snapshot stdlib.snapshot analysis.m
runmat --snapshot stdlib.snapshot
```

## Remote Projects

Remote commands connect the CLI to a RunMat server project. They are useful for hosted workspaces, shared project filesystems, and remote data layouts that should be mounted into local execution.

Authenticate first:

```bash
runmat login

# or explicitly specify the server URL

runmat login --server https://api.runmat.com
```

For automation, pass an API token:

```bash
runmat login \
  --server https://api.runmat.com \
  --api-key "$RUNMAT_API_KEY" \
  --project <project-id>
```

Alternatively, use environment variables to set the server URL and API token:

| Variable | Use |
| --- | --- |
| `RUNMAT_CONFIG` | Runtime config path. |
| `RUNMAT_SERVER_URL` | Remote server URL. |
| `RUNMAT_API_KEY` | Remote API token. |
| `RUNMAT_ORG_ID` | Default remote org. |
| `RUNMAT_PROJECT_ID` | Default remote project. |

List and select projects:

```bash
runmat org list
runmat project list
runmat project select <project-id>
```

Use the project filesystem:

```bash
runmat project fs ls /data
runmat project fs read /data/input.mat --output input.mat
runmat project fs write /data/input.mat ./input.mat
runmat project fs mkdir /data/results --recursive
runmat project fs rm /data/old.mat
```

The top-level `fs` command is a shorthand for the project filesystem namespace:

```bash
runmat fs ls /data
```

Run a script loaded from the remote filesystem:

```bash
runmat remote run /scripts/analysis.m
runmat remote run /scripts/analysis.m --project <project-id>
```

`remote run` reads the source from the selected remote project, installs the remote filesystem provider for the run, and executes the script locally with the current runtime configuration.

Remote filesystem commands also cover file history, manifest history, snapshots, retention policy, and git-style project sync. Use command help for the full tree:

```bash
runmat project fs --help
runmat project retention --help
runmat fs --help
```

## Command Help

Every command and subcommand has built-in help.

```bash
runmat --help
runmat run --help
runmat config --help
runmat project fs --help
```

Use command help as the source of truth for exact flags in the installed version.
