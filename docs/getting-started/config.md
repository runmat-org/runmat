# Configuration Reference

RunMat reads project metadata and runtime settings from `runmat.toml` or `runmat.json`. User settings supply defaults for fields a project leaves unset; environment variables and CLI arguments can override them.

For a user-facing guide to project layout, source roots, dependencies, and named entrypoints, see [Projects](/docs/runtime/getting-started/projects). This page is the reference for manifest and runtime settings.

## Configuration Resolution Order

RunMat resolves configuration settings using a specific precedence. Lower-numbered levels are overridden by higher-numbered levels:

1. Built-in defaults.
2. User settings in `~/.config/runmat/settings-v1.json`, or the settings document supplied by the host. If that file is absent, native discovery checks `config.toml` and `config.json` in the same directory.
3. The nearest project manifest, discovered by walking upward from the project directory.
4. Explicit configuration files: `RUNMAT_CONFIG`, then the CLI's `--config` file when supplied.
5. Field-specific environment overrides, such as `RUNMAT_JIT_THRESHOLD`.
6. Individual CLI arguments, such as `--no-jit` or `--gc-preset`.

Tables merge recursively, arrays replace, and an omitted field inherits its lower-layer value. `runmat config show --provenance` reports each effective field's source. `runmat config user show`, `runmat config user set jit enabled --value false`, and `runmat config user reset jit enabled` inspect or change only the user layer; edits preserve unrelated fields and comments where supported. `RUNMAT_USER_SETTINGS` selects another user settings path.

Browser hosts supply a JSON user document instead of a home-directory file. Cluster runs carry only project settings and built-in defaults: neither the submitter's nor the worker's personal settings or environment become run policy. Worker resource limits and device assignments remain authoritative.

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

Desktop project settings are stored in the same `runmat.toml` or `runmat.json` document as package, source, test, and runtime configuration. Updating Desktop settings preserves unrelated sections and comments.

Legacy `.runmat` settings are migrated automatically into the project’s `runmat.toml` or `runmat.json` file.

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

## Runtime Reference

All runtime settings are under `[runtime]`. Runtime settings control the behavior of the RunMat runtime.

### `[runtime]`


| Key               | Type    | Default | Notes                                                                                    |
| ----------------- | ------- | ------- | ---------------------------------------------------------------------------------------- |
| `callstack_limit` | integer | `200`   | Session call-stack limit.                                          |
| `error_namespace` | string  | `""`    | Error ID namespace. Empty value is normalized at startup by [language compatibility mode](/docs/runtime/matlab-compatibility) (set to `RunMat` in `compat = "runmat"` mode). |
| `verbose`         | boolean | `false` | Enables verbose execution output.                                                        |


### `[runtime.language]`


| Key      | Type   | Default    | Allowed values               | Notes                        |
| -------- | ------ | ---------- | ---------------------------- | ---------------------------- |
| `compat` | string | `"runmat"` | `runmat`, `matlab`, `strict` | Language compatibility mode. |


See [MATLAB Language Compatibility](/docs/runtime/matlab-compatibility) for the compatibility modes and [MATLAB Language Extensions](/docs/runtime/getting-started/language-extensions) for the RunMat-only behavior controlled by those modes.

### `[runtime.foreign.mex]`

| Key | Type | Default | Allowed values | Notes |
| --- | --- | --- | --- | --- |
| `unmanifested` | string | `"isolate"` | `isolate`, `deny` | Runs supported unmanifested modules in a local extension-host process, or rejects them before startup. |
| `timeout_ms` | positive integer | unset | — | Terminates an isolated host when an invocation or lifecycle callback exceeds the configured duration. An unset value applies no automatic invocation deadline; clear and shutdown retain a 30-second teardown ceiling. Cancellation also terminates the host. |

Modules with a valid RunMat build manifest retain exact admission and run in the owning session. The `unmanifested` setting applies only to the separately checked binary-compatibility tier; it does not bypass architecture, suffix, or adapter-ABI checks.

A configured MEX deadline is rejected before loading a manifested in-process module: arbitrary native instructions cannot be safely preempted. The default has no invocation deadline. Supported isolated routes enforce their configured deadlines by terminating the host.

Use `[mex-artifacts.<name>]` for a RunMat-built MEX module that belongs to the project. The declaration names the module and its canonical sidecar manifest relative to the package root. RunMat freezes both files, validates their shared identity for the current target, and carries them through packages, native compilation, and remote execution:

```toml
[mex-artifacts.filters]
manifest = "mex/filters.mexa64.runmat.json"
module = "mex/filters.mexa64"
```

The table name must match the module name recorded by the manifest. The filename suffix remains platform-specific, so packages that support several native targets should select target-specific dependencies or artifacts rather than presenting one binary as portable. Browser resolution rejects packages that require MEX before execution.

### `[runtime.foreign.native]`

| Key | Type | Default | Allowed values | Notes |
| --- | --- | --- | --- | --- |
| `isolation` | string | `"process"` | `process`, `in_process` | Runs shared-library calls in a same-executable child host by default. `in_process` is an explicit trusted-code opt-in. |
| `timeout_ms` | positive integer | unset | — | Terminates the isolated native-library host when one invocation exceeds the configured duration. An unset value applies no automatic invocation deadline; cancellation still terminates the host. |
| `timeout` | table | unset | `{ secs = ..., nanos = ... }` | An exact-duration alternative to `timeout_ms`. The duration must be positive and `nanos` must be below 1,000,000,000. Specify only one timeout representation. |

Process isolation contains native crashes and hangs but does not reduce the library's operating-system permissions. Packaged standalone programs and distributed workers always use the process boundary. See [Native Library Interfaces](/docs/execution/native-libraries) for preparation, packaging, callbacks, and pointer lifetime behavior.

### `[runtime.foreign.java]`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `home` | path | unset | Uses this Java installation instead of `JAVA_HOME` or platform discovery. The path must identify a complete installation with a loadable JVM library. |
| `minimum_version` | positive integer | `8` | Rejects an older Java runtime before JVM startup. |
| `maximum_version` | positive integer | unset | Rejects a newer Java runtime before JVM startup. |
| `classpath` | array of paths | `[]` | Project-owned Java directories and JARs. These form the session's static project classpath; `javaaddpath` manages a separate dynamic layer. |
| `options` | array of strings | `[]` | JVM startup options. They are immutable after the process JVM starts. |

RunMat starts Java lazily on the first operation that needs the JVM. `jenv` reports the selected installation and load state without starting it, and `javaclasspath` can inspect the configured project and dynamic layers. A process can host only one JVM configuration; a later session must use the same installation and startup options once Java has started.

The Desktop host enables `javaObjectEDT` and `javaMethodEDT` for synchronous AWT Event Dispatch Thread calls. CLI, server, and browser sessions reject EDT operations explicitly. See [Java Interoperability](/docs/runtime/execution/java) for value conversion, classloader, callback, and EDT behavior.

Use `[java-artifacts.<name>]` for JARs that are part of the project rather than local machine configuration. Each entry has one project-relative `path`; RunMat freezes its exact bytes for package, native compilation, and remote execution:

```toml
[java-artifacts.analysis]
path = "lib/analysis.jar"
```

### `[runtime.foreign.python]`

| Key | Type | Default | Allowed values | Notes |
| --- | --- | --- | --- | --- |
| `executable` | path | unset | — | Selects a specific CPython executable. |
| `version` | string | unset | `major.minor` | Requests one CPython feature version. |
| `minimum_version` | string | `"3.9"` | `major.minor` | Rejects an older interpreter before startup. |
| `maximum_version` | string | unset | `major.minor` | Rejects a newer interpreter before startup. |
| `execution_mode` | string | `"in_process"` | `in_process`, `out_of_process` | Chooses direct in-process execution or an isolated child mode of the same `runmat` executable. |
| `exact_version` | table | unset | `{ major = ..., minor = ..., patch = ... }` | Pins the complete CPython version, in addition to the feature-version bounds. |
| `required_abi_tag` | string | unset | — | Requires the ABI tag reported by the selected interpreter. |
| `required_platform_tag` | string | unset | — | Requires the platform tag reported by the selected interpreter. |
| `module_paths` | array of paths | `[]` | — | Adds host-local import roots. Custom roots are not portable package artifacts; remote execution rejects paths it cannot authenticate and materialize. |

Use `[python-artifacts.<name>]` for a wheel that belongs to the project. `path` is a project-relative `.whl` file and `module` is its top-level import name:

```toml
[python-artifacts.analysis]
path = "python/analysis-1.0-py3-none-any.whl"
module = "analysis"
```

RunMat freezes and revalidates declared wheels, carries them through packages, native compilation, and remote execution, and materializes them into a session-owned import root. Native wheels remain target-specific. Browser/WASM products reject native CPython requirements before execution. See [Python Interoperability](/docs/runtime/execution/python) for value conversion, isolation, callbacks, datetime behavior, and packaging details.

The isolated extension host contains crashes and can be terminated on cancellation or timeout, but it is not an operating-system security sandbox. An admitted native module has the extension-host process's permissions. Use `unmanifested = "deny"` when only exact, manifest-bound RunMat artifacts should be allowed.

### `[runtime.jit]`


| Key                  | Type    | Default   | Notes                                   |
| -------------------- | ------- | --------- | --------------------------------------- |
| `enabled`            | boolean | `true`    | Enables JIT compilation.                |
| `threshold`          | positive integer | `10`      | Base activity threshold for native tiering. |
| `optimization_level` | string  | `"speed"` | `none`, `size`, `speed`, `aggressive`. |

For threshold N, native tiering uses N for generic compilation, 2N for specialization and 4N for loops. `aggressive` specializes at N. Runtime representation profiles and compilation budgets also govern eligibility; placement separately selects eligible implementations. See [JIT Compilation Pipeline](/docs/runtime/jit/pipeline).

`none`, `size` and `speed` select the native compiler's corresponding optimization policy; `aggressive` uses speed optimization with earlier specialization. Changing JIT policy creates a fresh compiler-cache generation without discarding active executors. Browser execution has no native JIT backend.


### `[runtime.gc]`


| Key             | Type    | Default | Notes                                                    |
| --------------- | ------- | ------- | -------------------------------------------------------- |
| `preset`        | string  | unset   | `low-latency`, `high-throughput`, `low-memory`, `debug`. |
| `young_size_mb` | integer | unset   | Young generation size override (MB).                     |
| `threads`       | integer | unset   | Explicit collector thread count; only `1` is supported. Other counts reject. |
| `collect_stats` | boolean | `false` | Retained statistics preference; existing collector counters are maintained independently of this flag. |

The collector is serial; this setting does not limit numerical kernels or FEA worker threads. Desktop GC configuration belongs to the process. Its first runtime selects the policy; a different later GC request requires a Desktop restart rather than silently changing another window's collector. Cluster workers also check GC settings against their admitted CPU and memory resources.


### `[runtime.accelerate]`


| Key                           | Type    | Default  | Notes                                                         |
| ----------------------------- | ------- | -------- | ------------------------------------------------------------- |
| `enabled`                     | boolean | `true`   | Enables acceleration subsystem.                               |
| `provider`                    | string  | `"wgpu"` | `auto`, `wgpu`, `inprocess`.                                  |
| `allow_inprocess_fallback`    | boolean | `true`   | Falls back to in-process provider if hardware provider fails. |
| `wgpu_power_preference`       | string  | `"auto"` | `auto`, `high-performance`, `low-power`.                      |
| `wgpu_force_fallback_adapter` | boolean | `false`  | Forces WGPU fallback adapter selection.                       |

Desktop retains a provider for each physical settings tuple and reuses it across sessions with matching settings. Runtime acceleration and fallback restrictions remain invocation-scoped. Disabling computation does not invalidate the owner of an existing GPU value: an authorized readback can still gather it, but cannot grant new device computation.


#### `[runtime.accelerate.auto_offload]`


| Key            | Type    | Default   | Notes                                                    |
| -------------- | ------- | --------- | -------------------------------------------------------- |
| `enabled`      | boolean | `true`    | Enables auto-offload planner.                            |
| `calibrate`    | boolean | `true`    | Enables the runtime's automatic calibration templates.    |
| `log_level`    | string  | `"trace"` | `off`, `info`, `trace`.                                  |

The former `profile_path` setting is removed; delete it from existing configuration files. Category-only GPU profile JSON and `RUNMAT_ACCEL_PROFILE` are not placement authorities. Matched measurements use the canonical placement session; `runmat accel-calibrate --refresh` refreshes the current provider/precision's calibration cache.


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
| `maximum_pending_opens` | positive integer | `64` | Maximum queued native-window open requests. |
| `maximum_windows` | positive integer | `64` | Maximum live native windows owned by the host. |
| `maximum_snapshot_bytes` | positive integer | `268435456` | Upper bound on the retained Figure scene batch admitted for native presentation. |
| `maximum_pending_callbacks` | positive integer | `1024` | Maximum queued graphics callback events. A full queue reports backpressure. |
| `maximum_callback_registrations` | positive integer | `1024` | Maximum callback registrations in a session's graphics queue. |

These live-GUI limits must be nonzero. They bound open requests, retained scene
batches and callback queues independently; they do not bound renderer/GPU or
whole-process memory. `maximum_snapshot_bytes` is separate from the serialized
export/replay limit `runtime.plotting.export.scene_budget_bytes`. Raise an
individual limit only when the application needs the corresponding capacity.


#### `[runtime.plotting.export]`


| Key          | Type    | Default | Notes                            |
| ------------ | ------- | ------- | -------------------------------- |
| `format`     | string  | `"png"` | `png`, `svg`, `pdf`, `html`.     |
| `dpi`        | integer | `300`   | Raster export DPI.               |
| `output_dir` | string  | unset   | Default export output directory. |
| `scene_budget_bytes` | integer | `8388608` | Maximum serialized figure-scene payload size used by Desktop/browser export and replay. |


### `[runtime.fea]`

FEA settings configure study execution and run artifact storage.

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `artifact_store` | string | `filesystem` | `filesystem` or `in_memory` FEA run store. |
| `artifact_root` | string | `"artifacts"` | Filesystem root used by FEA run artifacts and derived evidence roots. |
| `artifact_max_runs` | integer | unset | Optional global retained run limit. |
| `artifact_max_runs_per_kind` | integer | unset | Optional retained run limit per physics family. |
| `study_artifact_root` | string | `"artifacts/studies"` | Study validate, plan, run, and sweep evidence root. |

Example:

```toml
[runtime.fea]
artifact_store = "filesystem"
artifact_root = "artifacts"
artifact_max_runs = 1000
artifact_max_runs_per_kind = 100
study_artifact_root = "artifacts/studies"
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
| `file`  | string  | unset    | Native append-log path; unsupported by the browser host. |

Desktop's local file/filter selection is process-owned. Its owning window can change it; a different window with conflicting settings receives an explicit error. Collector export is separate from both local logging and RunMat product telemetry.

#### `[runtime.logging.otel]`

| Key | Default | Notes |
| --- | --- | --- |
| `enabled` | `false` | Enables export to the user's collector. |
| `endpoint` | `"http://localhost:4318"` | OTLP base URL; HTTP appends `/v1/traces` and `/v1/logs`. |
| `protocol` | `"http/protobuf"` | Native supports `http/protobuf` and `grpc`; browsers require HTTP and collector CORS permission. |
| `signals` | `["traces", "logs"]` | Requested signals. Metrics are unsupported and reject at startup. |
| `headers_from_env` | `"RUNMAT_OTEL_HEADERS"` | Name of the startup environment variable containing collector headers, not the headers themselves. Desktop supplies secrets from its keychain. |
| `service_name` | `"runmat"` | Service resource name. |
| `resource_attributes` | `{}` | String-valued resource attributes. |

Export uses bounded queues and counts dropped records instead of waiting for collector networking on the execution path. A process owns one collector configuration; incompatible later selections require a restart. The test-export operation waits for an actual collector acknowledgement, not only queue insertion. Logs carry trace/span IDs where a trace context exists. Never put authorization headers in project or user settings.

Standard `OTEL_EXPORTER_OTLP_*`, `OTEL_SERVICE_NAME` and `OTEL_RESOURCE_ATTRIBUTES` variables are supported. Per-signal HTTP endpoints are complete URLs. `RUNMAT_OTEL_ENDPOINT` is a shortcut that enables export; `RUNMAT_OTEL_ENABLED=false` explicitly disables it. Cluster runs do not carry personal collector settings or secrets; workers use an administrator-selected collector or none.


## Environment Variables

### Config Selection

- `RUNMAT_CONFIG`: absolute or relative path to `runmat.toml` / `runmat.json`
- `RUNMAT_USER_SETTINGS`: explicit native user settings file

Runtime fields also accept `RUNMAT_RUNTIME__<SECTION>__<FIELD>` overrides, for example `RUNMAT_RUNTIME__ACCELERATE__AUTO_OFFLOAD__ENABLED=false`. Values are JSON literals, with plain text accepted for strings. Arrays replace their lower-layer values.

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

[runtime.foreign.mex]
unmanifested = "isolate"
# timeout_ms = 300000

[runtime.foreign.native]
isolation = "process"
# timeout_ms = 300000

[runtime.foreign.java]
minimum_version = 8
# maximum_version = 21
# home = "/path/to/java/home"
classpath = []
options = []

[runtime.jit]
enabled = true

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
maximum_pending_opens = 64
maximum_windows = 64
maximum_snapshot_bytes = 268435456
maximum_pending_callbacks = 1024
maximum_callback_registrations = 1024

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
