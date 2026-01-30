RunMat Logging & Tracing Guide
=============================

Scope
-----
This document describes the runtime-side logging and tracing stack for the open-source `runmat` crates. It covers Rust logging/tracing usage, subscriber configuration, OTLP export, and the WebAssembly bridge surface.

Overview
--------
- **Crate:** `runmat-logging` centralizes structured logs/traces.
- **Macros:** use standard `tracing` and `log` macros; no project-specific macros are required.
- **Subscriber:** `runmat_logging::init_logging` installs a `tracing_subscriber` pipeline that:
  - Applies `EnvFilter` (driven by environment).
  - Bridges `log` to `tracing` via `tracing_log::LogTracer`.
  - Emits structured runtime log records via `LogBridgeLayer`.
  - Emits Chrome Trace–style events via `TraceBridgeLayer`.
  - Optionally exports OTLP (logs/traces) when enabled.
- **WASM bridge:** `runmat-wasm` exposes subscription APIs so JavaScript can receive runtime logs and trace events.

How to log in Rust
------------------
- Prefer `tracing` macros:
  - `tracing::error!`, `warn!`, `info!`, `debug!`, `trace!`
  - Spans for lifecycles and phases: `info_span!`, `debug_span!`
- Existing `log` macros (`log::info!`, etc.) are also supported; they are bridged into `tracing`.
- Avoid `println!` / `eprintln!` in non-test code; route everything through `tracing`/`log`.

Structured log schema
---------------------
`runmat-logging` emits `RuntimeLogRecord`:
- `ts`: nanoseconds since UNIX epoch (i64)
- `level`: TRACE | DEBUG | INFO | WARN | ERROR
- `target`: log target (module path)
- `message`: formatted log message
- `fields`: optional JSON map of structured fields
- `trace_id` / `span_id`: populated when a `tracing::Span` is active; uses OTEL context when available, otherwise a fallback span id.

Trace schema
------------
`TraceEvent` follows the Chrome Trace Event format (subset):
- `name`: event/span name
- `cat`: category
- `ph`: phase (`B`egin / `E`nd / `X` complete / `i` instant)
- `ts`: timestamp in microseconds
- `dur`: optional duration (microseconds) for complete events
- `trace_id` / `span_id`: captured from the active span when present

Key instrumentation (runtime)
-----------------------------
- Execute lifecycle: spans around top-level execute/dispatch.
- Interpreter: spans for frames/steps (debug-gated).
- Fusion/compile/JIT: spans around lower/compile/execute phases.
- GPU: spans for dispatches, buffer uploads/downloads, transfers, and render/compute passes (WGPU provider).

Initialization
--------------
Call once during runtime startup:
```rust
runmat_logging::init_logging(LoggingOptions {
    enable_otlp: false,
    enable_traces: true,
    pid: 0,
    default_filter: None,
});
```
This installs the subscriber globally. `LogTracer` is initialized first so any `log` invocations flow through `tracing`. In WASM builds, `runmat-wasm::init_logging_once` wraps this, supplies a `default_filter` of `debug`, and installs panic hooks before the subscriber is created.

`LoggingOptions`
----------------
- `enable_otlp`: toggle the optional OTLP exporter (requires the `otlp` cargo feature).
- `enable_traces`: attach the `TraceBridgeLayer` so `TraceEvent`s are produced.
- `pid`: process identifier for trace metadata.
- `default_filter`: optional `EnvFilter` expression used when neither `RUST_LOG` nor `RUNMAT_LOG` are set. All builds now fall back to `info` unless you override this value explicitly.

Environment controls
--------------------
- `RUST_LOG` / `RUNMAT_LOG`: standard `tracing_subscriber::EnvFilter` syntax.
  - Example: `RUST_LOG=info,runmat_ignition=debug,runmat_accelerate=trace`
- OTLP (optional, behind feature flags):
  - `RUNMAT_OTEL_EXPORT=1` (enable)
  - `RUNMAT_OTEL_ENDPOINT=<http(s) collector endpoint>`
  - `RUNMAT_OTEL_HEADERS` (optional) comma-separated key=value
If OTLP is disabled, logging remains local and still streams to the JS hook when registered.

WASM surface
------------
Provided by `runmat-wasm` (wasm-bindgen):
- `subscribeRuntimeLogs(callback: Fn(RuntimeLogRecord)) -> unsubscribe`
- `unsubscribeRuntimeLogs()`
- `subscribeTraceEvents(callback: Fn(Vec<TraceEvent>)) -> unsubscribe`
- `unsubscribeTraceEvents()`
- `handleInit` installs forwarders; `handleDispose` tears them down.

JS/host integration expectations
--------------------------------
- The host registers callbacks to receive runtime logs and trace events and can render or forward them as needed.
- Logs and traces are distinct channels; stdout is separate and unaffected.

Testing
-------
- `runmat-logging` unit tests validate that log and trace hooks fire and carry structured data.
- Runtime crates should rely on the shared macros; no crate-local logging helpers are necessary.

Best practices
--------------
- Use spans for lifecycle boundaries and costly operations (compile, dispatch, transfer).
- Prefer structured fields over string concatenation: `info!(count = n, "did work")`.
- Keep noisy debug tracing behind environment filters.
