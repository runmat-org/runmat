---
title: "Configuration Reference"
category: "Getting Started"
section: "1.5"
last_updated: "August 2, 2026"
---

# Configuration Reference

RunMat utilizes a hierarchical configuration system that manages project-level metadata, source organization, and runtime execution parameters. The configuration is primarily driven by a manifest file (`runmat.toml` or `runmat.json`) and can be overridden by environment variables and CLI arguments.

For a user-facing guide to project layout, source roots, dependencies, and named entrypoints, see [Projects](/docs/runtime/getting-started/projects). This page is the reference for manifest and runtime settings.

## Configuration Resolution Order

RunMat resolves configuration settings using a specific precedence. Lower-numbered levels are overridden by higher-numbered levels:

1. Built-in Defaults
2. Project Manifest: Settings found in `runmat.toml` or `runmat.json`. The system automatically discovers this file by walking up the directory tree from the source file being executed
3. Environment Variables: Variables such as `RUNMAT_CONFIG` or `RUNMAT_JIT_THRESHOLD`
4. CLI Arguments: Explicit flags passed to the `runmat` binary (e.g., `--no-jit` or `--gc-preset`)

## Example Project Manifest (`runmat.toml`)

```toml
[package]
name = "my-project"
version = "0.1.0"
runmat-version = ">=0.4.0"

[dependencies]
utils = { path = "../utils", version = "0.1.0" }

[sources]
roots = ["src"]

[entrypoints.hello]
path = "hello.m"

[runtime.language]
compat = "runmat"
```

You can then execute the project's `hello` entrypoint with:

```bash
runmat run hello
```

## Project Reference

Project sections describe package identity, source layout, dependencies, and named entrypoints.

### `[package]`


| Key              | Type   | Default  | Notes                                                      |
| ---------------- | ------ | -------- | ---------------------------------------------------------- |
| `name`           | string | required | Package identifier.                                        |
| `version`        | string | unset    | Package version metadata.                                  |
| `runmat-version` | string | unset    | Minimum RunMat version gate. Accepts `>=x.y.z` or `x.y.z`. |


### `[sources]`


| Key     | Type     | Default  | Notes                                                           |
| ------- | -------- | -------- | --------------------------------------------------------------- |
| `roots` | string[] | required | Source root directories, relative to the config file directory. |


### `[dependencies]`

Each dependency is keyed by alias.

```toml
[dependencies]
utils = { path = "../utils", version = "0.1.0" }
```


| Field      | Type   | Default | Notes                                                                  |
| ---------- | ------ | ------- | ---------------------------------------------------------------------- |
| `path`     | string | unset   | Local dependency path; mutually exclusive with remote source fields.   |
| `git`      | string | unset   | Credential-free HTTPS or SSH Git repository URL.                       |
| `rev`      | string | unset   | Exact Git commit selector.                                             |
| `tag`      | string | unset   | Mutable Git tag selector; mutually exclusive with `rev` and `branch`.  |
| `branch`   | string | unset   | Mutable Git branch selector.                                           |
| `subdir`   | string | `""`    | Package subdirectory inside a Git repository.                          |
| `project`  | string | unset   | RunMat Server project ID.                                              |
| `service`  | string | active Server | Credential-free HTTPS Server origin.                             |
| `snapshot` | string | `main`  | Server snapshot tag or exact `snap_...` ID.                             |
| `version`  | string | unset   | Version requirement or metadata for the dependency declaration.        |


### `[entrypoints.<name>]`

Define named targets that can be executed from CLI.

```toml
[entrypoints.main]
module = "app.main"
function = "main"
```

```toml
[entrypoints.batch]
path = "scripts/run_batch.m"
```


| Field      | Type   | Default | Notes                                             |
| ---------- | ------ | ------- | ------------------------------------------------- |
| `path`     | string | unset   | File target. `.m` is inferred when omitted; FEA study files should use `.fea`. |
| `module`   | string | unset   | Module path under source roots.                   |
| `function` | string | unset   | Function name for module target.                  |


Exactly one target mode is required: `path` or `module + function`.

Entrypoint CLI examples:

```bash
runmat run main
runmat run studies/bracket_static.fea
runmat benchmark main --iterations 25 --jit
```

## Desktop Project Reference

Desktop project behavior is stored in the same canonical `runmat.toml` or `runmat.json` document as package, source, test, and runtime configuration. RunMat parses, validates, migrates, and updates these settings through one Rust configuration authority in native, CLI, and WASM/browser hosts; an update preserves sections and comments owned by other subsystems.

When a legacy `.runmat` file is present, RunMat performs one finite promotion into the canonical project document. Existing canonical values win, missing legacy and unowned values are merged even when the destination is `runmat.json`, and `.runmat` is removed only after the canonical write is read back successfully; later reads and writes use only the canonical file.

```toml
[desktop.artifacts]
root = ".artifacts"

[desktop.run_history]
mode = "budgeted"
trace = true
logs = "all"

[desktop.script]
clear_workspace_before_run = true
clear_figures_before_run = true

[desktop.notebook]
on_error = "stop"
rerun_after_cancel = "remaining"
```

| Section and key | Type | Default | Allowed values / notes |
| --- | --- | --- | --- |
| `desktop.artifacts.root` | string | `".artifacts"` | Project-relative subdirectory; absolute paths, `..`, and overlap with RunMat configuration/internal-state paths are rejected. |
| `desktop.run_history.mode` | string | `"budgeted"` | `off`, `budgeted`, `full`. |
| `desktop.run_history.trace` | boolean | `true` | Persists the trace channel with retained runs. |
| `desktop.run_history.logs` | string | `"all"` | `off`, `errors`, `all`. |
| `desktop.script.clear_workspace_before_run` | boolean | `true` | Clears workspace state before a script run. |
| `desktop.script.clear_figures_before_run` | boolean | `true` | Clears figures before a script run. |
| `desktop.notebook.on_error` | string | `"stop"` | `stop`, `continue`. |
| `desktop.notebook.rerun_after_cancel` | string | `"remaining"` | `remaining`, `all`. |

Device and user-experience preferences are intentionally not project configuration. Command-window placement is stored per device; internal-artifact visibility and notebook workspace auto-restore are stored per user and project; background runtime diagnosis is explicit, default-off, account-level consent stored by RunMat Server. These preferences never alter the package graph, static analysis, type/shape analysis, or reproducible runtime configuration.

## Runtime Reference

All runtime settings are under `[runtime]`. Runtime settings control the behavior of the RunMat runtime.

### `[runtime]`


| Key               | Type    | Default | Notes                                                                                    |
| ----------------- | ------- | ------- | ---------------------------------------------------------------------------------------- |
| `callstack_limit` | integer | `200`   | Max retained call stack frames for diagnostics.                                          |
| `error_namespace` | string  | `""`    | Error ID namespace. Empty value is normalized at startup by [language compatibility mode](/docs/runtime/getting-started/compatability) (set to `RunMat` in `compat = "runmat"` mode). |
| `verbose`         | boolean | `false` | Enables verbose execution output.                                                        |


### `[runtime.language]`


| Key      | Type   | Default    | Allowed values               | Notes                        |
| -------- | ------ | ---------- | ---------------------------- | ---------------------------- |
| `compat` | string | `"runmat"` | `runmat`, `matlab`, `strict` | Language compatibility mode. |


See [MATLAB Language Compatability](/docs/runtime/getting-started/compatability) for more details on runtime language compatibility modes.

### `[runtime.jit]`


| Key                  | Type    | Default   | Notes                                   |
| -------------------- | ------- | --------- | --------------------------------------- |
| `enabled`            | boolean | `true`    | Enables JIT compilation.                |
| `threshold`          | integer | `10`      | Executions before JIT tiering triggers. |
| `optimization_level` | string  | `"speed"` | `none`, `size`, `speed`, `aggressive`.  |


### `[runtime.gc]`


| Key             | Type    | Default | Notes                                                    |
| --------------- | ------- | ------- | -------------------------------------------------------- |
| `preset`        | string  | unset   | `low-latency`, `high-throughput`, `low-memory`, `debug`. |
| `young_size_mb` | integer | unset   | Young generation size override (MB).                     |
| `threads`       | integer | unset   | GC worker thread override.                               |
| `collect_stats` | boolean | `false` | Enables GC statistics collection.                        |


### `[runtime.accelerate]`


| Key                           | Type    | Default  | Notes                                                         |
| ----------------------------- | ------- | -------- | ------------------------------------------------------------- |
| `enabled`                     | boolean | `true`   | Enables acceleration subsystem.                               |
| `provider`                    | string  | `"wgpu"` | `auto`, `wgpu`, `inprocess`.                                  |
| `allow_inprocess_fallback`    | boolean | `true`   | Falls back to in-process provider if hardware provider fails. |
| `wgpu_power_preference`       | string  | `"auto"` | `auto`, `high-performance`, `low-power`.                      |
| `wgpu_force_fallback_adapter` | boolean | `false`  | Forces WGPU fallback adapter selection.                       |


#### `[runtime.accelerate.auto_offload]`


| Key            | Type    | Default   | Notes                                                    |
| -------------- | ------- | --------- | -------------------------------------------------------- |
| `enabled`      | boolean | `true`    | Enables auto-offload planner.                            |
| `calibrate`    | boolean | `true`    | Enables calibration mode for planner profile generation. |
| `profile_path` | string  | unset     | Optional profile cache path.                             |
| `log_level`    | string  | `"trace"` | `off`, `info`, `trace`.                                  |


### `[runtime.plotting]`


| Key                     | Type    | Default  | Notes                                      |
| ----------------------- | ------- | -------- | ------------------------------------------ |
| `mode`                  | string  | `"auto"` | `auto`, `gui`, `headless`.                 |
| `force_headless`        | boolean | `false`  | Forces non-interactive rendering behavior. |
| `backend`               | string  | `"auto"` | `auto`, `wgpu`, `static`, `web`.           |
| `scatter_target_points` | integer | unset    | Optional scatter decimation target.        |
| `surface_vertex_budget` | integer | unset    | Optional surface vertex LOD budget.        |


#### `[runtime.plotting.gui]`


| Key         | Type    | Default | Notes                      |
| ----------- | ------- | ------- | -------------------------- |
| `width`     | integer | `1200`  | Default GUI window width.  |
| `height`    | integer | `800`   | Default GUI window height. |
| `vsync`     | boolean | `true`  | Enables VSync.             |
| `maximized` | boolean | `false` | Starts window maximized.   |


#### `[runtime.plotting.export]`


| Key          | Type    | Default | Notes                            |
| ------------ | ------- | ------- | -------------------------------- |
| `format`     | string  | `"png"` | `png`, `svg`, `pdf`, `html`.     |
| `dpi`        | integer | `300`   | Raster export DPI.               |
| `output_dir` | string  | unset   | Default export output directory. |
| `scene_budget_bytes` | integer | `8388608` | Maximum serialized figure-scene payload size used by Desktop/browser export and replay. |


### `[runtime.fea]`

FEA settings configure study execution, run artifact storage, geometry prep artifacts, and coupled-field artifact roots.

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `artifact_store` | string | `filesystem` | `filesystem` or `in_memory` FEA run store. |
| `artifact_root` | string | `"artifacts"` | Filesystem root used by FEA run artifacts and derived evidence roots. |
| `artifact_max_runs` | integer | unset | Optional global retained run limit. |
| `artifact_max_runs_per_kind` | integer | unset | Optional retained run limit per physics family. |
| `study_artifact_root` | string | `"artifacts/studies"` | Study validate, plan, run, and sweep evidence root. |
| `geometry_prep_artifact_root` | string | `"artifacts/geometry-prep"` | Geometry prep artifact root. |
| `geometry_prep_max_artifacts` | integer | unset | Optional global retained prep artifact limit. |
| `geometry_prep_max_artifacts_per_geometry` | integer | unset | Optional retained prep artifact limit per geometry id. |
| `geometry_prep_max_age_seconds` | integer | unset | Optional prep artifact age limit. |
| `geometry_prep_require_latest_revision` | boolean | unset | When true, prep-aware runs reject stale geometry revisions. |
| `thermo_field_artifact_root` | string | `"artifacts/thermo-fields"` | Thermo-field artifact root for coupled thermal paths. |

Example:

```toml
[runtime.fea]
artifact_store = "filesystem"
artifact_root = "artifacts"
artifact_max_runs = 1000
artifact_max_runs_per_kind = 100
study_artifact_root = "artifacts/studies"
geometry_prep_artifact_root = "artifacts/geometry-prep"
geometry_prep_max_artifacts = 500
geometry_prep_max_artifacts_per_geometry = 20
geometry_prep_max_age_seconds = 2592000
geometry_prep_require_latest_revision = true
thermo_field_artifact_root = "artifacts/thermo-fields"
```


### `[runtime.telemetry]`


| Key                     | Type    | Default                           | Notes                                                                         |
| ----------------------- | ------- | --------------------------------- | ----------------------------------------------------------------------------- |
| `enabled`               | boolean | `true`                            | Enables telemetry client.                                                     |
| `show_payloads`         | boolean | `false`                           | Echoes serialized payloads to stdout.                                         |
| `http_endpoint`         | string  | unset                             | Optional HTTP override. When unset, runtime uses built-in collector endpoint. |
| `udp_endpoint`          | string  | `"udp.telemetry.runmat.com:7846"` | UDP collector endpoint.                                                       |
| `queue_size`            | integer | `256`                             | Async telemetry queue size (minimum bounded internally).                      |
| `sync_mode`             | boolean | `false`                           | Sends telemetry synchronously on caller thread.                               |
| `drain_mode`            | string  | `"all"`                           | `none`, `all`.                                                                |
| `drain_timeout_ms`      | integer | `50`                              | Max drain wait on shutdown (capped internally).                               |
| `require_ingestion_key` | boolean | `true`                            | Disables telemetry if key is required but unavailable.                        |


### `[runtime.logging]`


| Key     | Type    | Default  | Notes                                                                     |
| ------- | ------- | -------- | ------------------------------------------------------------------------- |
| `level` | string  | `"warn"` | `error`, `warn`, `info`, `debug`, `trace`.                                |
| `debug` | boolean | `false`  | Forces debug logging path.                                                |
| `file`  | string  | unset    | Reserved log file path option (runtime currently logs to process logger). |


## Environment Variables

### Config Selection

- `RUNMAT_CONFIG`: absolute or relative path to `runmat.toml` / `runmat.json`

### Service/Auth

- `RUNMAT_API_KEY`
- `RUNMAT_SERVER_URL`
- `RUNMAT_ORG_ID`
- `RUNMAT_PROJECT_ID`

### Telemetry

- `RUNMAT_TELEMETRY_KEY` (ingestion key override)

### FEA

FEA runtime config is preferred. These environment variables are supported as fallbacks:

- `RUNMAT_FEA_ARTIFACT_STORE`
- `RUNMAT_FEA_ARTIFACT_ROOT`
- `RUNMAT_FEA_ARTIFACT_MAX_RUNS`
- `RUNMAT_FEA_ARTIFACT_MAX_RUNS_PER_KIND`
- `RUNMAT_FEA_STUDY_ARTIFACT_ROOT`
- `RUNMAT_GEOMETRY_PREP_ARTIFACT_ROOT`
- `RUNMAT_GEOMETRY_PREP_MAX_ARTIFACTS`
- `RUNMAT_GEOMETRY_PREP_MAX_ARTIFACTS_PER_GEOMETRY`
- `RUNMAT_GEOMETRY_PREP_MAX_AGE_SECONDS`
- `RUNMAT_GEOMETRY_PREP_REQUIRE_LATEST_REVISION`
- `RUNMAT_THERMO_FIELD_ARTIFACT_ROOT`

## Full Reference Example

```toml
[package]
name = "image-pipeline"
version = "0.1.0"
runmat-version = ">=0.4.0"

[sources]
roots = ["src", "lib"]

[dependencies]
utils = { path = "../utils", version = "0.1.0" }

[entrypoints.main]
module = "app.main"
function = "main"

[entrypoints.batch]
path = "scripts/run_batch.m"

[runtime]
callstack_limit = 200
error_namespace = "RunMat"
verbose = false

[runtime.language]
compat = "runmat"

[runtime.jit]
enabled = true
threshold = 10
optimization_level = "speed"

[runtime.gc]
preset = "low-latency"
young_size_mb = 128
threads = 8
collect_stats = false

[runtime.accelerate]
enabled = true
provider = "wgpu"
allow_inprocess_fallback = true
wgpu_power_preference = "auto"
wgpu_force_fallback_adapter = false

[runtime.accelerate.auto_offload]
enabled = true
calibrate = true
profile_path = ".runmat/auto_offload.json"
log_level = "trace"

[runtime.plotting]
mode = "auto"
force_headless = false
backend = "auto"
scatter_target_points = 250000
surface_vertex_budget = 400000

[runtime.plotting.gui]
width = 1200
height = 800
vsync = true
maximized = false

[runtime.plotting.export]
format = "png"
dpi = 300
output_dir = "artifacts/figures"

[runtime.fea]
artifact_store = "filesystem"
artifact_root = "artifacts"
artifact_max_runs = 1000
artifact_max_runs_per_kind = 100
study_artifact_root = "artifacts/studies"
geometry_prep_artifact_root = "artifacts/geometry-prep"
geometry_prep_max_artifacts = 500
geometry_prep_max_artifacts_per_geometry = 20
geometry_prep_max_age_seconds = 2592000
geometry_prep_require_latest_revision = true
thermo_field_artifact_root = "artifacts/thermo-fields"

[runtime.telemetry]
enabled = true
show_payloads = false
udp_endpoint = "udp.telemetry.runmat.com:7846"
queue_size = 256
sync_mode = false
drain_mode = "all"
drain_timeout_ms = 50
require_ingestion_key = true

[runtime.logging]
level = "warn"
debug = false
```
