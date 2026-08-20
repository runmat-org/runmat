---
title: "Command Line Interface"
category: "Getting Started"
section: "1.3"
last_updated: "July 31, 2026"
---

# RunMat Command Line Interface (CLI)

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
| `!cmd` | Run `cmd` in the platform shell and print stdout/stderr. |

Shell escapes are local CLI REPL behavior. A line such as `!pwd` runs `pwd` through the host shell without submitting it to the RunMat parser. Non-zero shell exits are reported at the prompt and do not close the REPL. `.m` scripts and other hosts do not treat leading `!` as a shell escape.

The REPL also accepts piped input:

```bash
printf "1 + 1\n" | runmat repl
printf "!pwd\n1 + 1\n" | runmat repl
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

`runmat run` also runs `.fea` study and parametric sweep files:

```bash
runmat run studies/bracket_static.fea
runmat run --json studies/bracket_static.fea
```

The default `.fea` output is a concise run summary with the `run_id`, quality status, evidence path, and a `fea.results("<run_id>")` post-processing hint. Use `--json` when automation needs structured output.

RunMat also resolves configured project entrypoints. If a project has `runmat.toml` with an entrypoint named `main`, this works:

```toml
% runmat.toml
[entrypoints.main]
path = "src/main.m"
```

```bash
runmat run main
```

See [Projects](/docs/runtime/getting-started/projects) for project layout and entrypoint configuration.

For relative source paths, RunMat can infer a missing `.m` extension:

```bash
runmat run src/main
```

Execution uses the same session pipeline as other hosts: parse, lower, compile, run, emit streams, update workspace, and report structured diagnostics.

## Compile

Use `runmat compile` to produce a standalone executable for the current host:

```bash
runmat compile main.m -o main
./main
```

Compilation uses the same project/package composition, HIR, MIR, static analysis, builtin catalog, and Native IR as normal execution. Functions in the entry file and configured project source roots are compiled with their stable program identities; list stable cross-directory sources under `[sources].roots` so they are available to static composition. A runtime-only `addpath` cannot retroactively add source code to an already linked executable.

The default `--policy=native-specialized` emits target-native code and links the matching RunMat runtime archive embedded in the `runmat` binary. The result does not require a separate RunMat installation or SDK directory at runtime. `--optimization=none|size|speed` controls native object optimization; speed is the default.

The final host link uses a supported system linker and the native development libraries behind enabled runtime features, such as HDF5. Install those platform dependencies and expose their standard library search paths to the linker; RunMat supplies its own matching execution runtime and does not require a separate RunMat SDK.

Use `--policy=closed-world` when every runtime target can be proven. RunMat links only the exact reachable catalog-backed builtin bindings, uses ordinary archive extraction and platform dead stripping, and omits parser, compiler, VM, JIT, and object-emission code from the resulting executable. The small HIR/MIR operation schema required to decode and verify Native IR remains part of the runtime. If a call can reach an unknown target, a dynamically extensible builtin, or a builtin without an exact native binding, compilation fails with a specific diagnostic; use `native-specialized` when that broader runtime discovery is intentional.

The `dynamic-runtime` and `portable` policy names reserve distinct product contracts and fail clearly until their complete implementations are available: `dynamic-runtime` requires an embedded frontend and dynamic source loader, while `portable` produces a target-independent artifact instead of a host executable. RunMat does not silently approximate one policy with another.

Temporary object, archive, and response-file inputs are private and removed after linking. Use `--keep-temps` for linker diagnosis, `--linker PATH` to select an explicit supported system linker driver, and `--force` to replace an output. Forced replacement preserves the previous executable until the new link succeeds.

Use `--explain-link` to see the exact program functions, builtins, classes, method boundaries, providers, extensions, and runtime families retained by compilation, together with the direct, finite-dynamic, or unknown reason for each edge. Closed-world explanations also name each retained builtin binding variant and its stable native symbol. Use `--link-plan-json PATH` to write the same deterministic report for CI or tooling. The JSON includes the program, runtime archive, builtin catalog, target, policy, capability, and reachability identities used by that compilation; `--force` is required to replace an existing report.

The compiler validates that the embedded runtime matches the current target, native ABI, schema, RunMat version, runtime identity, and builtin catalog before linking. A source build made with ordinary `cargo build` intentionally lacks that large embedded archive; use the two-phase build helper described in [Build System](/docs/runtime/development/build-system) when developing the standalone workflow.

## Check

Use `runmat check` before running a `.m` script or `.fea` study:

```bash
runmat check analysis.m
runmat check --path ./toolbox analysis.m
runmat check -D warnings analysis.m
runmat check --json analysis.m
runmat check studies/bracket_static.fea
runmat check --json studies/bracket_static.fea
```

For `.m` files, check runs the same parser, HIR and MIR lowering, static analysis, source lookup, and compile validation used by editor tooling without executing the script. It reports syntax and semantic errors, proven type or shape incompatibilities, and function calls that cannot be resolved from builtins, the file, or the configured project sources. `--path DIRECTORY` adds an explicit MATLAB lookup root for the check and may be repeated.

Dynamic MATLAB behavior is reported without being rejected by default. For example, a function that is not present in the static source catalog produces a warning, and a call after `addpath` identifies that path mutation as the reason the final target must be selected and loaded at runtime. This warning describes a supported dynamic execution boundary, not an execution failure; use `[sources].roots` or `--path` when the target should participate in static cross-file analysis. Warnings leave the command successful, while errors return a nonzero exit code. Use `-D warnings` (or `-D warning`) when CI should also return nonzero for any warning.

The default output is human-readable and includes diagnostic codes, source locations, related causal locations, notes, and help. For `.m` files, `--json` emits the stable `schema_version: 1` envelope with an explicit `outcome` (`clean`, `warnings`, or `failed`), per-domain analysis completeness, structured diagnostics with byte and line/column spans, and summary counts. A failed check still emits that JSON payload before returning nonzero.

For `.fea` files, check loads geometry, resolves selectors, validates the study or sweep, and builds the solve plan without running the solver.

FEA JSON mode returns structured validation and plan payloads for CI and tooling.

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
| `--color MODE` | Control ANSI styling for human output (auto | always | never). |
| `--debug` | Enable debug logging. |
| `--log-level LEVEL` | Set log verbosity. |
| `--verbose` | Print more execution detail. |
| `--offline` | Resolve packages only from locally available content. |
| `--locked` | Require an existing, current `runmat.lock`. |
| `--frozen` | Require the lock and prohibit network access or lock mutation. |
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

## Clusters and Remote Jobs

Cluster administration, enrollment, and durable encrypted jobs use the active Server, organization, and project credentials:

```bash
runmat cluster list
runmat cluster create --name workstation-pool --queue default
runmat cluster enroll CLUSTER_ID --ttl-seconds 900
runmat cluster nodes CLUSTER_ID
runmat cluster node-state CLUSTER_ID NODE_ID draining

runmat job submit analysis.m --cluster CLUSTER_ID --trust-identity SHA256_FINGERPRINT --detach
runmat job list
runmat job show RUN_ID
runmat job attach RUN_ID
runmat job cancel RUN_ID

runmat job recovery keygen --output runmat-recovery.json
runmat job recovery configure --org ORG_ID --key runmat-recovery.json
runmat job recovery recover RUN_ID --project PROJECT_ID --key runmat-recovery.json
```

Every cluster command and every durable job observation/mutation supports `--json`. JSON mode emits one complete stable API object or page and never includes ANSI escapes; human list mode remains stable tab-separated output. Enrollment output contains a single-use secret, so avoid shell history and logs and prefer JSON-to-secret-store automation when scripting it.

The node agent can run in the foreground for diagnosis or install its native systemd, launchd, or Windows service. Generate a dry-run plan before changing a host:

```bash
runmat cluster join --server https://api.runmat.com --runmat /usr/local/bin/runmat service install --dry-run
sudo runmat cluster join --server https://api.runmat.com --runmat /usr/local/bin/runmat service install
```

Service installation persists only non-secret configuration. Enrollment credentials remain in the private state directory, service removal preserves that identity for safe reinstall, and retiring a host requires revoking the node before deleting its state. See `runmat cluster join --help` for platform paths and the foreground enrollment flow.

Organization recovery is optional. The CLI generates and retains the private key locally, sends only its validated public recipient to the Server policy API, and decrypts authorized terminal results or diagnostics on the custodian machine. Once configured, every new submission must carry an envelope for the exact active fingerprint. Keep rotated private keys for the full artifact-retention period. See [Remote Execution](/docs/runtime/execution/remote) for customer-node, hosted-node, browser, draining, and recovery workflows.

## Packages

Package resolution is available directly and is also applied automatically by run, REPL, check, benchmark, and bytecode workflows:

```bash
runmat package resolve
runmat package fetch
runmat package update
runmat package tree
runmat package why DEPENDENCY
runmat package vendor
runmat package cache status
runmat package cache gc
runmat package cache prune
```

`resolve` creates or refreshes `runmat.lock`; `fetch` fills missing immutable cache content without selecting a newer locked commit; and `update` is the explicit operation that may advance a branch or tag. `tree` and `why` project the same resolved graph used by execution and static analysis. `vendor` writes project-local dependency copies plus a workspace-root `runmat-vendor.json`; frozen execution requires and verifies that record for live path dependencies. See [Projects](/docs/runtime/getting-started/projects) for Git syntax, lock modes, browser behavior, cache recovery, and vendoring.

## Test Projects

`runmat test` discovers MATLAB-style script, function, and class tests semantically, selects them without executing test bodies, freezes the package graph and source revision, and runs the resulting immutable plan:

```bash
runmat test
runmat test tests/solver --name convergence --tag fast
runmat test --jobs 4 --isolation process --report junit
runmat test --coverage --coverage-format lcov
runmat test --cluster CLUSTER_ID --project PROJECT_ID --trust-identity SHA256_FINGERPRINT --max-workers 8
```

Runtime and test dependency groups are resolved together before discovery. The default native isolation is a fresh killable process; `session` and `none` are explicit weaker modes. Reports, captured output, diagnostics, artifacts, retries, cancellation, and coverage are projections of one deterministic event/result stream. Human, JSON, JUnit, and TAP reports are supported, as are JSON, LCOV, Cobertura, and HTML coverage reports. Exact options and defaults are available from `runmat test --help`.

The same portable discovery, plan, coordinator, result, and coverage authorities back `runtests`, browser Web Worker execution, and Desktop. Browser hosts use fresh dedicated Web Workers as their strongest available isolation and preserve selected source and fixture bytes in an immutable worker-local filesystem snapshot.

Remote tests use the same coordinator through the general execution scheduler and encrypted execution control/data plane. `--cluster` overrides `[test.cluster].profile`, `--queue` overrides `[test.cluster].queue`, and `--max-workers` caps concurrent remote fixture groups. A pinned `--trust-identity` is required before protected plan, source, result, event, artifact, or coverage content is encrypted to the admitted endpoint. The Server sees only the coarse execution metadata needed for admission, leases, routing, retention, and billing; it cannot decrypt test content or detailed results.

Remote jobs carry an exact compiled program artifact and frozen package identities inside the encrypted execution bundle, without source bytes or a project handoff. Drivers and workers validate the artifact, target, package graph, and source revision identities and execute it without reconstructing a source tree. Remote tests retain the separate source-project closure they need for fixtures, dynamic test discovery, and source-aware reporting; workers validate and materialize those verified logical objects into a private read-only source root. Package resolution and private-package decryption happen only on the submitting client in both cases, so remote workers need no Git, Server-project, registry, or package-decryption credentials and cannot silently select a different dependency.

## Color and Terminal Output

RunMat uses restrained ANSI styling for human-readable diagnostics, help, headings, status messages, and summaries. The default `--color=auto` mode checks stdout and stderr independently, styles only streams connected to capable interactive terminals, and stays plain when output is redirected, `TERM=dumb`, or a non-empty `NO_COLOR` value is present.

Use the global color option before or after a subcommand:

```bash
runmat --color=never check analysis.m
runmat check analysis.m --color=never
runmat --color=always check analysis.m | less -R
```

An explicit `--color=always` or `--color=never` overrides the environment. Without an explicit option, a non-empty `NO_COLOR` disables color; `CLICOLOR=0` also disables it; and `CLICOLOR_FORCE` or `FORCE_COLOR` can request color for eligible human output. `NO_COLOR` takes precedence over those environment force variables. An empty `NO_COLOR` value is treated as unset.

Structured and byte-oriented output remains plain even under `--color=always`. This includes JSON, TOML configuration, bytecode, stable tab-separated remote listings, telemetry payloads, and raw remote file contents. RunMat also leaves MATLAB stdout and stderr, displayed MATLAB values, and REPL shell-command output unchanged.

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
