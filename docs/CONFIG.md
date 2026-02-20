# RunMat Configuration

RunMat provides a modern, explicit configuration system designed for clarity
and portability. It supports multiple file formats, sane defaults, rich
environment variable overrides, and precise CLI flag precedence — a major step
up from legacy largely process-flag driven approach.

This document is the definitive reference for all configuration options.

**Applicability:** Configuration files, environment variables, and the `runmat config` subcommands apply to the **CLI** and to tools that use the config loader (e.g. the LSP). The **browser sandbox** does not read `.runmat` or env vars; it uses built-in defaults. Use the CLI or desktop app for file-based and environment-based configuration.

## Overview

Configuration sources (highest to lowest precedence):
1. Command-line flags
2. Environment variables (`RUNMAT_*`)
3. Configuration files (`.runmat`, `.runmat.yaml`, `runmat.config.json`, …)
4. Built-in defaults

Loading follows `runmat_config::ConfigLoader::load()` (in the `runmat-config` crate) which:
- Searches for a config file (see discovery paths below)
- Loads and parses it (YAML/JSON/TOML)
- Applies environment variable overrides
- CLI flags are applied by `runmat` after loading

## Discovery paths

Checked in order; first existing file is loaded:

1. Explicit path via `RUNMAT_CONFIG`
2. Current directory candidates:
   - `.runmat` (preferred; TOML syntax)
   - `.runmat.yaml`, `.runmat.yml`, `.runmat.json`, `.runmat.toml`
   - `runmat.config.yaml`, `runmat.config.yml`, `runmat.config.json`, `runmat.config.toml`
3. Home directory:
   - `~/.runmat`, `~/.runmat.yaml`, `~/.runmat.yml`, `~/.runmat.json`
   - `~/.config/runmat/config.yaml`, `config.yml`, `config.json`
4. System-wide (Unix):
   - `/etc/runmat/config.yaml`, `config.yml`, `config.json`

If nothing exists, defaults are used.

## File formats

RunMat supports YAML, JSON and TOML. Examples below encode the same config.

### TOML (.runmat)
```toml
[runtime]
timeout = 600
verbose = true
snapshot_path = "./stdlib.snapshot"

[jit]
enabled = true
threshold = 25
optimization_level = "speed"

[gc]
preset = "low-latency"
young_size_mb = 64
threads = 4
collect_stats = true

[kernel]
ip = "127.0.0.1"

[logging]
level = "warn"
debug = false

[plotting]
mode = "auto"
force_headless = false
backend = "auto"

[packages]
enabled = true

[[packages.registries]]
name = "runmat"
url = "https://packages.runmat.com"

[packages.dependencies]
# Resolve from registry
"linalg-plus" = { source = "registry", version = "^1.2" }
# Git dependency
"viz-tools" = { source = "git", url = "https://github.com/acme/viz-tools", rev = "main" }
# Local path during development
"my-local-ext" = { source = "path", path = "../my-local-ext" }
```

## Language configuration

Language-specific toggles live under the `[language]` table. Today it exposes the compatibility mode that controls whether MATLAB command syntax (`hold on`, `grid on`, etc.) is accepted:

```toml
[language]
compat = "matlab" # default
# compat = "strict"
```

- `matlab`: allows the curated set of command-style verbs documented in `/docs/language`, rewriting them into explicit calls before parsing.
- `strict`: disables command syntax entirely; scripts must call functions explicitly (e.g., `hold("on")`). Recommended for new codebases.

The CLI (`runmat`), native runtime, WASM runtime, and both LSP implementations read this setting automatically. Hosts can still override it via environment variables or LSP initialization options, but `.runmat` is treated as the source of truth.

### YAML (.runmat.yaml)
```yaml
runtime:
  timeout: 600
  verbose: true
  snapshot_path: ./stdlib.snapshot

jit:
  enabled: true
  threshold: 25
  optimization_level: speed

gc:
  preset: low-latency
  young_size_mb: 64
  threads: 4
  collect_stats: true

kernel:
  ip: 127.0.0.1
  key: null
  ports: null

logging:
  level: info
  debug: false
  file: null

plotting:
  mode: auto
  force_headless: false
  backend: auto
  gui:
    width: 1280
    height: 800
    vsync: true
    maximized: false
  export:
    format: png
    dpi: 200
    output_dir: ./plots
    jupyter:
      output_format: auto
      enable_widgets: true
      enable_static_fallback: true
      widget:
        client_side_rendering: true
        server_side_streaming: false
        cache_size_mb: 64
        update_fps: 30
        gpu_acceleration: true
      static_export:
        width: 800
        height: 600
        quality: 0.9
        include_metadata: true
        preferred_formats: [widget, png, svg]
      performance:
        max_render_time_ms: 16
        progressive_rendering: true
        lod_threshold: 10000
        texture_compression: true
```

### JSON (runmat.config.json)
```json
{
  "runtime": {"timeout": 600, "verbose": true, "snapshot_path": "./stdlib.snapshot"},
  "jit": {"enabled": true, "threshold": 25, "optimization_level": "speed"},
  "gc": {"preset": "low-latency", "young_size_mb": 64, "threads": 4, "collect_stats": true},
  "kernel": {"ip": "127.0.0.1", "key": null, "ports": null},
  "logging": {"level": "info", "debug": false, "file": null},
  "plotting": {
    "mode": "auto", "force_headless": false, "backend": "auto",
    "gui": {"width": 1280, "height": 800, "vsync": true, "maximized": false},
    "export": {
      "format": "png", "dpi": 200, "output_dir": "./plots",
      "jupyter": {
        "output_format": "auto", "enable_widgets": true, "enable_static_fallback": true,
        "widget": {"client_side_rendering": true, "server_side_streaming": false, "cache_size_mb": 64, "update_fps": 30, "gpu_acceleration": true},
        "static_export": {"width": 800, "height": 600, "quality": 0.9, "include_metadata": true, "preferred_formats": ["widget", "png", "svg"]},
        "performance": {"max_render_time_ms": 16, "progressive_rendering": true, "lod_threshold": 10000, "texture_compression": true}
      }
    }
  }
}
```

### TOML (runmat.config.toml)
```toml
[runtime]
timeout = 600
verbose = true
snapshot_path = "./stdlib.snapshot"

[jit]
enabled = true
threshold = 25
optimization_level = "speed"

[gc]
preset = "low-latency"
young_size_mb = 64
threads = 4
collect_stats = true

[kernel]
ip = "127.0.0.1"

[logging]
level = "warn"
debug = false

[plotting]
mode = "auto"
force_headless = false
backend = "auto"

[plotting.gui]
width = 1280
height = 800
vsync = true
maximized = false

[plotting.export]
format = "png"
dpi = 200
output_dir = "./plots"

[plotting.export.jupyter]
output_format = "auto"
enable_widgets = true
enable_static_fallback = true

[plotting.export.jupyter.widget]
client_side_rendering = true
server_side_streaming = false
cache_size_mb = 64
update_fps = 30
gpu_acceleration = true

[plotting.export.jupyter.static_export]
width = 800
height = 600
quality = 0.9
include_metadata = true
preferred_formats = ["widget", "png", "svg"]

[plotting.export.jupyter.performance]
max_render_time_ms = 16
progressive_rendering = true
lod_threshold = 10000
texture_compression = true
```

## Configuration schema (by module)

Below reflects the Rust types in `crates/runmat-config/src/lib.rs`. Defaults shown in
parentheses.

### runtime
- `timeout: u64` (300)
- `verbose: bool` (false)
- `snapshot_path: Path` (none)

### jit
- `enabled: bool` (true)
- `threshold: u32` (10)
- `optimization_level: one of {none,size,speed,aggressive}` (speed)

### gc
- `preset: one of {low-latency,high-throughput,low-memory,debug}` (none)
- `young_size_mb: usize` (none)
- `threads: usize` (none)
- `collect_stats: bool` (false)

### kernel
- `ip: String` ("127.0.0.1")
- `key: String` (none)
- `ports` (optional): `shell|iopub|stdin|control|heartbeat: u16` (none)

### logging
- `level: one of {error,warn,info,debug,trace}` (info)
- `debug: bool` (false)
- `file: Path` (none)

### plotting
- `mode: one of {auto,gui,headless,jupyter}` (auto)
- `force_headless: bool` (false)
- `backend: one of {auto,wgpu,static,web}` (auto)
- `gui` (optional):
  - `width: u32` (1200)
  - `height: u32` (800)
  - `vsync: bool` (true)
  - `maximized: bool` (false)
- `export` (optional):
  - `format: one of {png,svg,pdf,html}` (png)
  - `dpi: u32` (300)
  - `output_dir: Path` (none)
  - `jupyter` (optional):
    - `output_format: one of {widget,png,svg,base64,plotlyjson,auto}` (auto)
    - `enable_widgets: bool` (true)
    - `enable_static_fallback: bool` (true)
    - `widget` (optional):
      - `client_side_rendering: bool` (true)
      - `server_side_streaming: bool` (false)
      - `cache_size_mb: u32` (64)
      - `update_fps: u32` (30)
      - `gpu_acceleration: bool` (true)
    - `static_export` (optional):
      - `width: u32` (800)
      - `height: u32` (600)
      - `quality: f32` (0.9)
      - `include_metadata: bool` (true)
      - `preferred_formats: list<output_format>` ([widget, png, svg])
    - `performance` (optional):
      - `max_render_time_ms: u32` (16)
      - `progressive_rendering: bool` (true)
      - `lod_threshold: u32` (10000)
      - `texture_compression: bool` (true)

### packages
- `enabled: bool` (true)
- `registries: list<registry>` (defaults to the official `runmat` registry)
- `dependencies: map<string, package>` (empty)

Note: Package manager features are not yet released. The schema is shared to help early adopters prepare;
the `runmat pkg` commands will print a “coming soon” message until the first release lands.

Registry schema:
- `name: string`
- `url: string`

Package schema (tagged union by `source`):
- `{ source = "registry", version: string, registry?: string, features?: list<string>, optional?: bool }`
- `{ source = "git", url: string, rev?: string, features?: list<string>, optional?: bool }`
- `{ source = "path", path: string, features?: list<string>, optional?: bool }`

## Environment variables (overrides)

All `RUNMAT_*` variables map onto the above fields. Notable ones:

- Runtime: `RUNMAT_TIMEOUT`, `RUNMAT_VERBOSE`, `RUNMAT_SNAPSHOT_PATH`
- JIT: `RUNMAT_JIT_ENABLE`, `RUNMAT_JIT_DISABLE`, `RUNMAT_JIT_THRESHOLD`, `RUNMAT_JIT_OPT_LEVEL`
- GC: `RUNMAT_GC_PRESET`, `RUNMAT_GC_YOUNG_SIZE`, `RUNMAT_GC_THREADS`, `RUNMAT_GC_STATS`
- Plotting: `RUNMAT_PLOT_MODE`, `RUNMAT_PLOT_HEADLESS`, `RUNMAT_PLOT_BACKEND`
  Jupyter static export fallbacks: `RUNMAT_PLOT_JUPYTER_FORCE_CPU_EXPORT`,
  `RUNMAT_PLOT_JUPYTER_ALLOW_HEADLESS_GPU`
- Logging: `RUNMAT_DEBUG`, `RUNMAT_LOG_LEVEL`
- Kernel: `RUNMAT_KERNEL_IP`, `RUNMAT_KERNEL_KEY`

Boolean parsing accepts `1/0`, `true/false`, `yes/no`, `on/off`, `enable/disable`.

For Jupyter `png`/`base64` output, RunMat prefers a CPU placeholder export in CI and
headless Linux environments by default to avoid unstable GPU-driver paths. Set
`RUNMAT_PLOT_JUPYTER_ALLOW_HEADLESS_GPU=1` to force the GPU export path in those
environments.

### Acceleration provider (RunMat Accelerate)

Provider-specific env vars are read by the WGPU backend and fusion code:

- `RUNMAT_WG` (u32)
  - Global compute workgroup size used in WGSL at module creation.
    Applies to elementwise kernels and fused kernels. Default: `512`.
- `RUNMAT_MATMUL_TILE` (u32)
  - Square tile size used by matmul kernels.
    Default: `16`.
- `RUNMAT_REDUCTION_WG` (u32)
  - Default reduction workgroup size when call sites opt into provider defaults
    (passing `0`). Default: `512`.
- `RUNMAT_TWO_PASS_THRESHOLD` (usize)
  - Controls when two-pass reductions are used (per-slice length threshold).
  - Default: `1024`.
- `RUNMAT_DEBUG_PIPELINE_ONLY` (bool)
  - If set, provider may compile pipelines and skip buffer/dispatch paths to
    isolate driver issues during development.
- `RUNMAT_PIPELINE_CACHE_DIR` (path)
  - Overrides the on-disk pipeline cache directory. Defaults to the OS cache
    directory (e.g., `$XDG_CACHE_HOME/runmat/pipelines` or platform equivalent),
    falling back to `target/tmp/wgpu-pipeline-cache-<device>`.

> **Note:** `RUNMAT_WG`, `RUNMAT_MATMUL_TILE`, and `RUNMAT_REDUCTION_WG` are
> automatically clamped to the adapter's compute limits
> (`max_compute_workgroup_size_*`, `max_compute_invocations_per_workgroup`) to
> prevent invalid pipelines on DX12/Metal/Vulkan backends. The adjusted values
> are logged during provider initialization.

## Precedence & merging

1. Start from built-in defaults.
2. Merge config file if found (full or partial trees are fine).
3. Apply environment variables per field.
4. Apply CLI flags (final say).

Any field not specified remains at its previous value (default or earlier layer).

## Generating and validating configs

```sh
# Print a sample config to stdout
runmat --generate-config

# Write a sample config to file
runmat config generate -o .runmat.yaml

# Validate a config file
runmat config validate .runmat.yaml

# Show the current effective configuration (human-readable YAML)
runmat config show
```

## Practical examples

### 1) Developer laptop (fast iteration)
```yaml
# .runmat.yaml
jit:
  enabled: true
  threshold: 10
  optimization_level: speed
gc:
  preset: low-latency
plotting:
  mode: auto
  gui: { width: 1280, height: 800, vsync: true }
logging:
  level: info
```

```sh
runmat repl --verbose
```

### 2) CI / headless server
```yaml
plotting:
  mode: headless
gc:
  collect_stats: true
logging:
  level: warn
```

```sh
RUNMAT_JIT_DISABLE=1 runmat run tests/current_feature_test.m
```

### 3) Teaching lab machines
```yaml
runtime:
  timeout: 120
jit:
  enabled: true
  threshold: 5
gc:
  preset: low-memory
```

```matlab
% simple_intro.m
x = 1 + 2
A = [1, 2; 3, 4]
```

```sh
runmat run simple_intro.m
```

## Why this is better than MATLAB's configuration model

- Multiple file formats (YAML/JSON/TOML) with the same schema.
- Clear, documented precedence (flags > env > files > defaults).
- Explicit, typed configuration with sensible defaults and enums.
- First-class plotting/JIT/GC settings in config (not only runtime flags).
- Built-in generators/validators and human-readable dumps via `config show`.
- Portable configs you can commit, review, and diff.

## Troubleshooting

- Use `runmat config paths` to see where files are searched.
- Use `runmat info` to view effective configuration and environment.
- If a config file fails to parse, the CLI prints a precise error with file
  path and format (YAML/JSON/TOML) context.
