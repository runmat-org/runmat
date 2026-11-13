# User-owned OpenTelemetry export

This facility sends runtime logs and traces to the user's collector. It is
independent of RunMat product telemetry (`[telemetry]`): changing either switch
never changes the other. Unlike product telemetry, runtime logs may contain
application messages and fields; choose a trusted collector.

```toml
[logging.otel]
enabled = true
endpoint = "https://otel.example.com:4318"
protocol = "http/protobuf" # native also supports "grpc" (normally port 4317)
signals = ["traces", "logs"]
headers_from_env = "RUNMAT_OTEL_HEADERS"
service_name = "runmat"
resource_attributes = { "deployment.environment" = "dev" }
```

Metrics are not implemented; requesting `metrics` returns an explicit startup
error. Empty signal lists and unsupported protocols also fail explicitly.

## Native host contract

Pass the effective layered configuration as `otel::OtelOptions` to
`configure_otel(options, StartupHeaders(keychain_headers))` before installing the
subscriber. `init_logging` with `enable_otlp: true` attaches the configured layer.
Hosts that own their subscriber attach `otel_layer()` and
`runtime_bridge_layer(...)` instead. The latter forwards the same runtime events
to OTLP logs; those records carry the real OpenTelemetry trace/span context.
Do not register a second exporter/subscriber for the same process.
Product hosts that install their subscriber before opening the first project
attach `retained_otel_layer()` permanently instead. It retains SDK context even
before selecting a collector; spans born before selection are not exported.
Do not put the OpenTelemetry layer inside `reload::Layer`: its context downcast
must remain visible for correlation. Equal repeat selections reuse the exporter;
any enabled/disabled/settings conflict returns `OpenTelemetryRestartRequired`.

Hosts owning `log::Log` use `init_logging_with_host_logger` and forward original
records through `forward_log_record` (with `log_metadata_enabled` for filtering).
The CLI uses one such owner: it delegates terminal rendering to its unchanged
env_logger sink and forwards the same records to tracing. The terminal and
subscriber filters are independent, so terminal-only filtering cannot suppress
collector logs. This is a logger bridge, not a side effect of a formatter.

### Native CLI file output

```toml
[logging]
level = "info"
debug = false
file = "runmat.log"
```

The CLI opens `logging.file` in append mode before initializing collectors or
execution. Relative paths use the CLI process working directory; parent
directories must already exist. Opening failure is a startup error, not a
silently disabled sink. Terminal formatting, machine-readable stdout and the
single `log::Log` owner are unchanged. `debug=true` selects the debug filter;
otherwise the typed `level` selects its corresponding filter.

`init_logging_with_host_logger_and_file(options, owned_file)` adds one non-ANSI
formatted sink inside the existing native subscriber: original bridged log
records, tracing events and enabled span creation/close records reach the file.
It does not consume the runtime log hook, install another logger, or start an
exporter. An occupied global subscriber is rejected explicitly rather than
silently putting configured file logging on a competing thread-local subscriber.
Existing non-file initializers retain their legacy behavior.

File I/O is synchronous and has no extra queue. A write/flush failure is latched
and emits one non-recursive stderr warning without echoing the failed event.
Hosts call `LoggingGuard::flush_file_result()` at their result boundary; the CLI
does this after collector shutdown. A command that otherwise succeeds returns
an error if its configured file failed. When user code also fails, its original
error and cause chain remain primary and the file error is reported alongside
it. Drop still performs a best-effort flush. Files contain the same structured
runtime data as logging generally; existing secret-redaction obligations apply.
Hidden environment-only workers do not inherit a personal file sink implicitly.

`configure_otel` expects already-layered options; it does not reapply generic
environment overrides after the config loader. Environment-only hosts use
`OtelOptions::default().resolve_environment()` first. `RUNMAT_OTEL_ENDPOINT`
enables export as a shortcut; generic `OTEL_EXPORTER_OTLP_ENDPOINT` also enables
environment-only export. The RunMat shortcut wins the generic endpoint when
both are present. `OTEL_EXPORTER_OTLP_PROTOCOL`, `OTEL_SERVICE_NAME`, and
`OTEL_RESOURCE_ATTRIBUTES` are supported.

At exporter startup the standard per-signal
`OTEL_EXPORTER_OTLP_{TRACES,LOGS}_{ENDPOINT,PROTOCOL,HEADERS,TIMEOUT}` variables
override their generic counterparts. HTTP generic/config endpoints are base
URLs: `/v1/traces` and `/v1/logs` are appended. Per-signal HTTP endpoint variables
are complete URLs and are used unchanged. gRPC endpoints are used unchanged.
Timeouts are standard **milliseconds**, including `OTEL_EXPORTER_OTLP_TIMEOUT`.
Header/resource lists use comma-separated `key=value` pairs with percent-encoded
values (e.g. `authorization=Bearer%20secret`). Invalid values return a redacted
startup error.

Headers resolve from standard environment values, then the environment variable
named by `headers_from_env`, then host-supplied keychain headers (highest).
`StartupHeaders` has no serialization implementation and redacts its debug output.
Never put actual header secrets in project/user config, export receipts, or logs.
Do not pass credentials in endpoint userinfo. HTTP redirects are disabled so a
collector redirect cannot forward authorization headers to another origin.
Collector error bodies/messages and endpoint URLs are excluded from diagnostics.

The process owns one bounded queue (default 2048 records, maximum 4096), one
dedicated I/O thread/runtime and one transport per selected signal. Recording
only uses `try_send`: a full/disconnected queue drops the record and increments
`otel_stats().dropped_records`, without waiting for networking. Failed batches
increment `failed_batches`; only the first failure prints a warning. Collector
failures never fail a run. Exporters do not retry indefinitely.
Queued and in-flight encoded records share a strict 4 MiB reservation budget,
released on export/drop. Individual protobuf records are limited to 64 KiB;
messages to 8 KiB, individual fields to 4 KiB, resources to 16 KiB, and startup
headers to 8 KiB/16 entries. Bounds are checked before exporter cloning/encoding;
the log visitor also bounds formatting on the recording thread. SDK span
attribute/event/link counts are bounded; oversized finished spans are dropped
before protobuf conversion. Batches are limited to 512 KiB and responses are
streamed with a 64 KiB limit. Queue accounting excludes the separately bounded
transport batch/protobuf decode workspace and the SDK's in-progress spans.

Call `flush_otel` / `shutdown_otel_result` from a blocking host task before exit;
both await collector acknowledgements with bounded waits. CLI `OtelGuard` calls
the shutdown wrapper automatically. Settings' test button calls
`emit_test_otel` on a blocking host task: it creates an actual correlated test
span and OTLP log and waits for both collector responses. Disabled signals,
queue overflow, HTTP/gRPC failure, invalid protobuf, timeout and collector partial
rejection return `Err`, not a false queued-success receipt. Receipts contain only
counts. Success means collector acceptance, not downstream storage/indexing.
Each test command owns its own acknowledgement, so concurrent tests/flushes
cannot consume another test's failure. A restored collector can be tested without
a stale preflush failure preventing the new test. Ordinary flush/shutdown remain
an error if any prior background batch failed (those records were not delivered).
Zero rejected records with a collector warning are accepted and counted in
`collector_warnings`; raw warning text is never exposed. Individual HTTP/gRPC
operations have a whole-operation timeout (1–30000 ms); host test/flush/shutdown
waits are bounded to 15 seconds and return an explicit timeout on a backlog.

## Browser/worker host contract

Build `runmat-logging` with `browser-otlp`, call
`browser_otel::configure_browser_otel` before subscriber installation, and attach
`browser_otel_layer()` plus `runtime_bridge_layer(...)` (or use `init_logging`).
The browser uses the same SDK span model and generated OTLP protobuf messages,
an asynchronous bounded queue, and Fetch with credentials omitted. It supports
HTTP/protobuf only; selecting gRPC is an explicit error. Startup headers are
runtime-only host input; no OS keychain or process environment is assumed.
The browser queue is event-driven and performs no idle timer polling.

The WASM host resolves `runtimeSettings` before installing its subscriber and
reads `openTelemetryHeaders` separately from the raw host startup object; secrets
never enter serializable/debuggable session settings. The public SDK exposes
`sendOpenTelemetryTest`, `flushOpenTelemetry`, `shutdownOpenTelemetry`, and
`openTelemetryStats`. Equal repeated session configurations reuse the exporter.
Changing collector settings/headers, enabling export after a previously installed
unconfigured subscriber, or using a shut-down exporter returns an explicit
`OpenTelemetryRestartRequired` error instead of silently retaining old settings.
Initialize the session before subscribing streams or creating standalone renderers
when collector export is required. Session disposal does not shut down the shared
exporter; shutdown is a runtime-wide operation.

The collector must allow the app's exact origin through CORS, permit `POST` and
the `content-type`/configured authorization headers in preflight, and use HTTPS
when the app runs on HTTPS (mixed-content protection still applies). The app's
Content Security Policy must permit the collector in `connect-src`. Fetch runs
in both window and worker contexts. Network/TLS/CORS rejections produce a clear
redacted error; browsers do not expose enough detail to distinguish those causes.
HTTP statuses and protobuf partial rejections are checked exactly as on native.

Await `emit_test_browser_otel`, `flush_browser_otel`, or `shutdown_browser_otel`.
Never block the browser thread waiting for export. Hosts must flush before
terminating a worker or closing the runtime; abrupt tab/worker termination cannot
guarantee delivery. Queue/failure statistics are exposed by `browser_otel_stats`.

## Worker policy

Local child workers inherit explicit startup environment as before. Cluster
workers must use administrator-controlled collector settings, not the submitter's
personal user document or keychain headers; collector secrets must not travel in
project bundles or run manifests. Organization-level collector administration is
future work. With no administrator collector configured, remote workers export
neither user logs nor user traces. The caller is responsible for supplying that
policy at its worker startup boundary.
