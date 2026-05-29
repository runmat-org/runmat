---
title: "Telemetry"
category: "Development"
section: "14.5"
last_updated: "May 28, 2026"
---

# Telemetry

RunMat telemetry exists to answer a small set of product and runtime health questions: whether installs are succeeding by OS and architecture, how often acceleration and JIT execution are active, and where parser, compiler, runtime, GPU, or fusion failures happen often enough to prioritize.

The telemetry contract is intentionally narrow. Runtime events carry bounded metadata, counters, and error classes. They do not carry source code, tensor values, file contents, file paths, environment variables, shell history, stack traces, or workspace data.

## Event Model

Runtime telemetry is emitted as two process-level events:

| Event | Meaning |
| --- | --- |
| `runtime.run.started` | A CLI, REPL, benchmark, install, or host session started. |
| `runtime.run.finished` | The run finished, including duration, success state, counters, and optional provider telemetry. |

Both events use the same envelope shape:

| Field | Meaning |
| --- | --- |
| `event_label` | Canonical event name. |
| `uuid` | Event UUID. |
| `cid` | Anonymous client ID when available. |
| `session_id` | Runtime session ID. |
| `os`, `arch`, `release` | Platform and release metadata. |
| `run_kind` | `script`, `repl`, `benchmark`, or `install`. |
| `payload` | Event-specific runtime details. |

The started payload contains the JIT and acceleration enablement state. The finished payload adds duration, success, JIT usage, execution counters, optional bounded failure fields, and optional provider telemetry.

Failure fields are stable buckets:

| Field | Values |
| --- | --- |
| `runtime.failure.stage` | `parser`, `hir`, `compile`, `runtime`, or `unknown`. |
| `runtime.failure.code` | Stable diagnostic or runtime error identifier. |
| `runtime.failure.has_span` | Whether a diagnostic span was available. |
| `runtime.failure.host` | `cli`, `wasm`, `kernel`, or `desktop`. |
| `runtime.failure.component` | Optional bounded component bucket. |

Example finished payload:

```json
{
  "event_label": "runtime.run.finished",
  "uuid": "63b4b9a8-5f2b-4cef-95c1-c6e2b8f6c04c",
  "cid": "eac98648-3b42-41c7-a887-7452dd08cbf0",
  "session_id": "9e2a9d9f-4e37-4090-a1fb-96cb2c6f9f3a",
  "run_kind": "script",
  "os": "macos",
  "arch": "aarch64",
  "release": "0.4.1",
  "payload": {
    "duration_us": 4412,
    "success": true,
    "jit_enabled": true,
    "jit_used": true,
    "accelerate_enabled": true,
    "counters": {
      "total_executions": 1,
      "jit_compiled": 1,
      "interpreter_fallback": 0
    },
    "gpu_wall_ns": 1850000,
    "gpu_dispatches": 12,
    "gpu_upload_bytes": 262144,
    "gpu_download_bytes": 0,
    "fusion_cache_hits": 8,
    "fusion_cache_misses": 1
  }
}
```

## Client Identity

The CLI stores a random client ID at:

```text
~/.runmat/telemetry_id
```

On Windows the file lives under the profile directory:

```text
%USERPROFILE%\.runmat\telemetry_id
```

The ID is a random GUID. It is used to count sessions without using personal identifiers. Deleting the file causes the CLI to mint a new ID the next time telemetry is enabled.

## CLI Configuration

Runtime telemetry is configured through `runmat.toml` under `[runtime.telemetry]`.

```toml
[runtime.telemetry]
enabled = true
show_payloads = false
http_endpoint = ""
udp_endpoint = "udp.telemetry.runmat.com:7846"
queue_size = 256
sync_mode = false
drain_mode = "all"
drain_timeout_ms = 50
require_ingestion_key = true
```

| Key | Meaning |
| --- | --- |
| `enabled` | Enables runtime telemetry for the CLI session. |
| `show_payloads` | Prints each serialized payload before delivery. |
| `http_endpoint` | Optional HTTP endpoint override. Empty means the built-in collector endpoint. |
| `udp_endpoint` | Optional UDP endpoint override. Empty disables UDP delivery. |
| `queue_size` | Bounded async queue size. |
| `sync_mode` | Sends on the caller thread instead of a background worker. |
| `drain_mode` | `all` waits for queued events at shutdown; `none` exits without waiting. |
| `drain_timeout_ms` | Maximum shutdown wait, capped internally. |
| `require_ingestion_key` | Disables delivery when no ingestion key is available. |

Official builds can embed an ingestion key at compile time with `RUNMAT_TELEMETRY_KEY`. Source builds without that key do not deliver to the hosted collector when `require_ingestion_key = true`.

For local transparency, set:

```toml
[runtime.telemetry]
show_payloads = true
```

Payloads are printed to stdout before delivery. This does not disable delivery; it only shows what would be sent.

## Delivery

The CLI uses a background worker by default. It enqueues telemetry off the execution path and drains the queue when the process exits. The default drain timeout is 50 ms, with an internal maximum of 5 seconds.

Set `drain_mode = "none"` for runs where exit latency matters more than flushing telemetry:

```toml
[runtime.telemetry]
drain_mode = "none"
```

Delivery can use HTTP, UDP, or both, depending on configuration. The default HTTP endpoint is:

```text
https://api.runmat.com/v1/t
```

The default UDP endpoint is:

```text
udp.telemetry.runmat.com:7846
```

HTTP requests include the ingestion key in the `x-telemetry-key` header when one is available. Delivery failures are logged at debug level and do not fail the CLI run.

## Opt Out

Use the persisted runtime configuration to disable CLI telemetry:

```toml
[runtime.telemetry]
enabled = false
```

For browser hosts, pass `telemetryConsent: false` to `initRunMat`.

For source builds, leaving `RUNMAT_TELEMETRY_KEY` unset and keeping `require_ingestion_key = true` prevents delivery to the hosted collector. To remove the local anonymous ID, delete `~/.runmat/telemetry_id`.

Telemetry can be re-enabled by setting `enabled = true`, passing `telemetryConsent: true`, or building with an ingestion key when hosted delivery is intended.
