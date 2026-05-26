# RunMat Configuration

This document is the reference for `runmat.toml` / `runmat.json`.

## Config Resolution

RunMat loads configuration in this order:

1. Built-in defaults
2. Config file:
   - `RUNMAT_CONFIG` (explicit path)
   - nearest `runmat.toml` / `runmat.json` by walking up from cwd
   - user config at `~/.config/runmat/config.toml` / `~/.config/runmat/config.json`
3. CLI flags

CLI flags override file values.

## File Contract

- Supported extensions: `.toml`, `.json`
- A single file can contain both project and runtime config.
- Unknown keys are rejected for runtime config sections.

## Project Reference

Project sections describe package identity, source layout, dependencies, and named entrypoints.

### `[package]`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `name` | string | required | Package identifier. |
| `version` | string | unset | Package version metadata. |
| `runmat-version` | string | unset | Minimum RunMat version gate. Accepts `>=x.y.z` or `x.y.z`. |

### `[sources]`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `roots` | string[] | required | Source root directories, relative to the config file directory. |

### `[dependencies]`

Each dependency is keyed by alias.

```toml
[dependencies]
utils = { path = "../utils", version = "0.1.0" }
```

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `path` | string | unset | Local dependency path. Required for local composition today. |
| `version` | string | unset | Version metadata for dependency declaration. |

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

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `path` | string | unset | File target (`.m` extension inferred if omitted). |
| `module` | string | unset | Module path under source roots. |
| `function` | string | unset | Function name for module target. |

Exactly one target mode is required: `path` or `module + function`.

Entrypoint CLI examples:

```bash
runmat run main
runmat benchmark main --iterations 25 --jit
```

## Runtime Reference

All runtime settings are under `[runtime]`.

### `[runtime]`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `callstack_limit` | integer | `200` | Max retained call stack frames for diagnostics. |
| `error_namespace` | string | `""` | Error ID namespace. Empty value is normalized at startup by language compatibility mode. |
| `verbose` | boolean | `false` | Enables verbose execution output. |
| `snapshot_path` | string | unset | Optional snapshot file to preload. |

### `[runtime.language]`

| Key | Type | Default | Allowed values | Notes |
| --- | --- | --- | --- | --- |
| `compat` | string | `"runmat"` | `runmat`, `matlab`, `strict` | Language compatibility mode. |

### `[runtime.jit]`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | boolean | `true` | Enables JIT compilation. |
| `threshold` | integer | `10` | Executions before JIT tiering triggers. |
| `optimization_level` | string | `"speed"` | `none`, `size`, `speed`, `aggressive`. |

### `[runtime.gc]`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `preset` | string | unset | `low-latency`, `high-throughput`, `low-memory`, `debug`. |
| `young_size_mb` | integer | unset | Young generation size override (MB). |
| `threads` | integer | unset | GC worker thread override. |
| `collect_stats` | boolean | `false` | Enables GC statistics collection. |

### `[runtime.accelerate]`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | boolean | `true` | Enables acceleration subsystem. |
| `provider` | string | `"wgpu"` | `auto`, `wgpu`, `inprocess`. |
| `allow_inprocess_fallback` | boolean | `true` | Falls back to in-process provider if hardware provider fails. |
| `wgpu_power_preference` | string | `"auto"` | `auto`, `high-performance`, `low-power`. |
| `wgpu_force_fallback_adapter` | boolean | `false` | Forces WGPU fallback adapter selection. |

#### `[runtime.accelerate.auto_offload]`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | boolean | `true` | Enables auto-offload planner. |
| `calibrate` | boolean | `true` | Enables calibration mode for planner profile generation. |
| `profile_path` | string | unset | Optional profile cache path. |
| `log_level` | string | `"trace"` | `off`, `info`, `trace`. |

### `[runtime.plotting]`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `mode` | string | `"auto"` | `auto`, `gui`, `headless`. |
| `force_headless` | boolean | `false` | Forces non-interactive rendering behavior. |
| `backend` | string | `"auto"` | `auto`, `wgpu`, `static`, `web`. |
| `scatter_target_points` | integer | unset | Optional scatter decimation target. |
| `surface_vertex_budget` | integer | unset | Optional surface vertex LOD budget. |

#### `[runtime.plotting.gui]`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `width` | integer | `1200` | Default GUI window width. |
| `height` | integer | `800` | Default GUI window height. |
| `vsync` | boolean | `true` | Enables VSync. |
| `maximized` | boolean | `false` | Starts window maximized. |

#### `[runtime.plotting.export]`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `format` | string | `"png"` | `png`, `svg`, `pdf`, `html`. |
| `dpi` | integer | `300` | Raster export DPI. |
| `output_dir` | string | unset | Default export output directory. |

### `[runtime.telemetry]`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | boolean | `true` | Enables telemetry client. |
| `show_payloads` | boolean | `false` | Echoes serialized payloads to stdout. |
| `http_endpoint` | string | unset | Optional HTTP override. When unset, runtime uses built-in collector endpoint. |
| `udp_endpoint` | string | `"udp.telemetry.runmat.com:7846"` | UDP collector endpoint. |
| `queue_size` | integer | `256` | Async telemetry queue size (minimum bounded internally). |
| `sync_mode` | boolean | `false` | Sends telemetry synchronously on caller thread. |
| `drain_mode` | string | `"all"` | `none`, `all`. |
| `drain_timeout_ms` | integer | `50` | Max drain wait on shutdown (capped internally). |
| `require_ingestion_key` | boolean | `true` | Disables telemetry if key is required but unavailable. |

### `[runtime.logging]`

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `level` | string | `"warn"` | `error`, `warn`, `info`, `debug`, `trace`. |
| `debug` | boolean | `false` | Forces debug logging path. |
| `file` | string | unset | Reserved log file path option (runtime currently logs to process logger). |

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

## Full Example (`runmat.toml`)

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
