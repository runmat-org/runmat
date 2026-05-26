# RunMat Configuration

RunMat reads configuration from a project file named `runmat.toml` (or `runmat.json` if you prefer JSON).  
This file defines project structure and runtime behavior.

## How RunMat Finds Configuration

RunMat resolves configuration in this order:

1. Built-in defaults
2. A config file:
   - Path provided by `RUNMAT_CONFIG`
   - Otherwise, nearest `runmat.toml` or `runmat.json` found by walking up from the current directory
   - Otherwise, user config at `~/.config/runmat/config.toml` or `~/.config/runmat/config.json`
3. CLI flags

CLI flags take precedence over file values.

## Project Configuration

Project configuration describes package identity, source roots, dependencies, and entrypoints.

### `[package]`

- `name` (required)
- `version` (optional)
- `runmat-version` (optional minimum RunMat version)

Example:

```toml
[package]
name = "image-pipeline"
version = "0.1.0"
runmat-version = ">=0.4.0"
```

### `[sources]`

- `roots` (required list of source directories)

```toml
[sources]
roots = ["src", "lib"]
```

### `[dependencies]`

Dependencies are keyed by alias.

- `path` (local dependency path)
- `version` (optional version metadata)

```toml
[dependencies]
utils = { path = "../utils", version = "0.1.0" }
```

### `[entrypoints.<name>]`

Define named entrypoints for scripts or module functions.

Path-based:

```toml
[entrypoints.batch]
path = "scripts/run_batch.m"
```

Module/function-based:

```toml
[entrypoints.main]
module = "app.main"
function = "main"
```

CLI usage with named entrypoints:

```bash
runmat run main
runmat benchmark main --iterations 25 --jit
```

`main` resolves through `[entrypoints.main]` in `runmat.toml`/`runmat.json`.

## Runtime Configuration

All runtime settings live under `[runtime]`.
For default values, see [Runtime Defaults Reference](#runtime-defaults-reference).

### Core runtime fields

- `callstack_limit`
- `error_namespace`
- `verbose`
- `snapshot_path`

```toml
[runtime]
callstack_limit = 200
error_namespace = "RunMat"
verbose = false
```

### `[runtime.language]`

- `compat = "runmat" | "matlab" | "strict"`

```toml
[runtime.language]
compat = "runmat"
```

### `[runtime.jit]`

- `enabled`
- `threshold`
- `optimization_level = "none" | "size" | "speed" | "aggressive"`

```toml
[runtime.jit]
enabled = true
threshold = 10
optimization_level = "speed"
```

### `[runtime.gc]`

- `preset = "low-latency" | "high-throughput" | "low-memory" | "debug"`
- `young_size_mb`
- `threads`
- `collect_stats`

```toml
[runtime.gc]
preset = "low-latency"
young_size_mb = 128
threads = 8
collect_stats = false
```

### `[runtime.accelerate]`

- `enabled`
- `provider = "auto" | "wgpu" | "inprocess"`
- `allow_inprocess_fallback`
- `wgpu_power_preference = "auto" | "high-performance" | "low-power"`
- `wgpu_force_fallback_adapter`

```toml
[runtime.accelerate]
enabled = true
provider = "wgpu"
allow_inprocess_fallback = true
wgpu_power_preference = "auto"
wgpu_force_fallback_adapter = false
```

#### `[runtime.accelerate.auto_offload]`

- `enabled`
- `calibrate`
- `profile_path`
- `log_level = "off" | "info" | "trace"`

```toml
[runtime.accelerate.auto_offload]
enabled = true
calibrate = true
profile_path = ".runmat/auto_offload.json"
log_level = "trace"
```

### `[runtime.plotting]`

- `mode = "auto" | "gui" | "headless"`
- `force_headless`
- `backend = "auto" | "wgpu" | "static" | "web"`
- `scatter_target_points`
- `surface_vertex_budget`

```toml
[runtime.plotting]
mode = "auto"
force_headless = false
backend = "auto"
scatter_target_points = 250000
surface_vertex_budget = 400000
```

#### `[runtime.plotting.gui]`

- `width`
- `height`
- `vsync`
- `maximized`

```toml
[runtime.plotting.gui]
width = 1200
height = 800
vsync = true
maximized = false
```

#### `[runtime.plotting.export]`

- `format = "png" | "svg" | "pdf" | "html"`
- `dpi`
- `output_dir`

```toml
[runtime.plotting.export]
format = "png"
dpi = 300
output_dir = "artifacts/figures"
```

### `[runtime.telemetry]`

- `enabled`
- `show_payloads`
- `http_endpoint`
- `udp_endpoint`
- `queue_size`
- `sync_mode`
- `drain_mode = "none" | "all"`
- `drain_timeout_ms`
- `require_ingestion_key`

```toml
[runtime.telemetry]
enabled = true
show_payloads = false
udp_endpoint = "udp.telemetry.runmat.com:7846"
queue_size = 256
sync_mode = false
drain_mode = "all"
drain_timeout_ms = 50
require_ingestion_key = true
```

### `[runtime.logging]`

- `level = "error" | "warn" | "info" | "debug" | "trace"`
- `debug`
- `file`

```toml
[runtime.logging]
level = "warn"
debug = false
```

## Environment Variables

### Config path

- `RUNMAT_CONFIG`  
  Absolute or relative path to a config file.

### Service/auth settings

- `RUNMAT_API_KEY`
- `RUNMAT_SERVER_URL`
- `RUNMAT_ORG_ID`
- `RUNMAT_PROJECT_ID`

## Complete Example (`runmat.toml`)

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

## Runtime Defaults Reference

If a runtime key is omitted, RunMat uses the default shown below.  
`unset` means the value stays absent unless explicitly configured.

| Key | Default |
| --- | --- |
| `runtime.callstack_limit` | `200` |
| `runtime.error_namespace` | `""` |
| `runtime.verbose` | `false` |
| `runtime.snapshot_path` | `unset` |
| `runtime.language.compat` | `"runmat"` |
| `runtime.jit.enabled` | `true` |
| `runtime.jit.threshold` | `10` |
| `runtime.jit.optimization_level` | `"speed"` |
| `runtime.gc.preset` | `unset` |
| `runtime.gc.young_size_mb` | `unset` |
| `runtime.gc.threads` | `unset` |
| `runtime.gc.collect_stats` | `false` |
| `runtime.accelerate.enabled` | `true` |
| `runtime.accelerate.provider` | `"wgpu"` |
| `runtime.accelerate.allow_inprocess_fallback` | `true` |
| `runtime.accelerate.wgpu_power_preference` | `"auto"` |
| `runtime.accelerate.wgpu_force_fallback_adapter` | `false` |
| `runtime.accelerate.auto_offload.enabled` | `true` |
| `runtime.accelerate.auto_offload.calibrate` | `true` |
| `runtime.accelerate.auto_offload.profile_path` | `unset` |
| `runtime.accelerate.auto_offload.log_level` | `"trace"` |
| `runtime.plotting.mode` | `"auto"` |
| `runtime.plotting.force_headless` | `false` |
| `runtime.plotting.backend` | `"auto"` |
| `runtime.plotting.scatter_target_points` | `unset` |
| `runtime.plotting.surface_vertex_budget` | `unset` |
| `runtime.plotting.gui.width` | `1200` |
| `runtime.plotting.gui.height` | `800` |
| `runtime.plotting.gui.vsync` | `true` |
| `runtime.plotting.gui.maximized` | `false` |
| `runtime.plotting.export.format` | `"png"` |
| `runtime.plotting.export.dpi` | `300` |
| `runtime.plotting.export.output_dir` | `unset` |
| `runtime.telemetry.enabled` | `true` |
| `runtime.telemetry.show_payloads` | `false` |
| `runtime.telemetry.http_endpoint` | `unset` |
| `runtime.telemetry.udp_endpoint` | `"udp.telemetry.runmat.com:7846"` |
| `runtime.telemetry.queue_size` | `256` |
| `runtime.telemetry.sync_mode` | `false` |
| `runtime.telemetry.drain_mode` | `"all"` |
| `runtime.telemetry.drain_timeout_ms` | `50` |
| `runtime.telemetry.require_ingestion_key` | `true` |
| `runtime.logging.level` | `"warn"` |
| `runtime.logging.debug` | `false` |
| `runtime.logging.file` | `unset` |

## Related

- [CLI Reference](/docs/cli) -- commands, flags, environment variables, and examples.
- [Browser Guide](/docs/desktop-browser-guide) -- the browser-based sandbox IDE.
- [GPU Residency and Precision](/docs/accelerate/gpu-behavior) -- GPU residency rules and acceleration environment variables.
