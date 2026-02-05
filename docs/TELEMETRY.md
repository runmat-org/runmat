# Telemetry

## Why is telemetry collected?

RunMat telemetry exists to prove that we are delivering value to our users (e.g. that users are using RunMat). Anonymous usage signals tell us:

- Whether installers are succeeding by OS/arch.
- How often acceleration is enabled versus falling back to CPU.
- Where JIT compilation or fusion fails so we can prioritize fixes.

We only collect the minimum data needed to answer those questions, and we use it solely to improve the CLI/runtime.

## What is being collected?

We separate installer events from runtime events and keep the schemas intentionally small.

### Installers

Each run emits at most one of each event: `install_start`, `install_complete`, `install_failed`. Fields:

- `event_label` (one of the above)
- `os`, `arch`, `platform` (e.g. `darwin`, `arm64`, `macos-aarch64`)
- `release` (RunMat version being installed)
- `method` (`powershell` or `shell`)
- `cid` (anonymous client id, see below)

### Runtime (CLI / REPL / scripts)

At most two events per process:

- `runtime_started`: `session_id`, `cid`, `run_kind` (`script`, `repl`, `benchmark`, `kernel`), `os`, `arch`, CLI version, whether acceleration/JIT are enabled.
- `runtime_finished`: everything above plus `duration_us`, `success`, stringified error class (never source code), JIT usage flag, execution counters, provider metadata (device name/vendor/backend only), and GPU telemetry (dispatch counts, wall time, bytes moved, cache hits/misses, fusion stats).

Example runtime payload:

```json
{
  "event_label": "runtime_finished",
  "cid": "eac98648-3b42-41c7-a887-7452dd08cbf0",
  "session_id": "9e2a9d9f-4e37-4090-a1fb-96cb2c6f9f3a",
  "run_kind": "script",
  "os": "macos",
  "arch": "aarch64",
  "payload": {
    "duration_us": 4412,
    "success": true,
    "jit_enabled": true,
    "jit_used": true,
    "accelerate_enabled": true,
    "provider": {
      "device": { "name": "NVIDIA RTX 4090", "vendor": "NVIDIA", "backend": "cuda" },
      "telemetry": {
        "gpu_dispatches": 12,
        "gpu_wall_ns": 1850000,
        "upload_bytes": 262144,
        "download_bytes": 0,
        "fusion_cache_hits": 8,
        "fusion_cache_misses": 1
      }
    }
  }
}
```

### Browser / WASM hosts

The browser bindings expose a `telemetryConsent` flag on `initRunMat({ … })`. It defaults to `true` so development builds behave like the CLI, but hosts can pass `false` to disable telemetry entirely. When consent is disabled:

- Analytics collectors must call `RunMatSession::telemetry_consent()` before emitting anything. The wasm crate already forwards the JS flag to the session, and the CLI mirrors the user’s `telemetry.enabled` configuration into the same field, so hosts can respect the user’s preference without scraping CLI output. Profiling summaries (execution time, GPU counters, etc.) are still returned to the caller so local dashboards remain functional even when analytics are disabled.
- Hosts that already have an analytics identifier (e.g., the PostHog CID in the web shell) can forward it via the `telemetryId` init option so any future runtime-level telemetry reuses the same client id. `session.telemetryClientId()` exposes the resolved value so shells can confirm what the runtime will emit.

This keeps local tooling (profilers, dashboards) honest: if the user opts out, we avoid gathering GPU telemetry in the first place and nothing leaves the process.

### Anonymous client id

Installers and the CLI store a random GUID in `~/.runmat/telemetry_id` (or `%USERPROFILE%\.runmat\telemetry_id`). It lets us count sessions without using personal identifiers. Delete the file to regenerate a new id.

### See exactly what’s sent

Set `RUNMAT_TELEMETRY_SHOW=1` to print every JSON payload to stderr before it is sent:

```bash
RUNMAT_TELEMETRY_SHOW=1 runmat script.m
```

This flag is meant for transparency; telemetry is still transmitted unless disabled via the opts below.

## What about sensitive data?

We never collect:

- Source code, tensors, file contents, or paths.
- Environment variables or shell history.
- Serialized stack traces or values that may contain user data.

Only aggregated counts and metadata listed above are sent. For broader privacy questions, contact team@dystr.com.

## Where does telemetry go?

Clients send UDP (best effort) to `udp.telemetry.runmat.com:7846` (legacy `udp.telemetry.runmat.org:7846` still supported) or HTTPS POSTs to `https://telemetry.runmat.com/ingest` (legacy `https://telemetry.runmat.org/ingest`).

- HTTPS traffic lands on a Cloud Run service listening at `https://telemetry.runmat.com/ingest` (the source code is available at `infra/worker/` in the GitHub repo for transparency), normalizes payloads, and forwards to an analytics service (PostHog and Google Analytics [GA4] are used by the RunMat team for analytics).
- The UDP path flows through a lightweight Google Cloud UDP load balancer into a managed instance group running the forwarder container (code under `infra/udp-forwarder/` in the repo). Each forwarder replays datagrams to the Cloud Run endpoint asynchronously so the CLI never blocks.

No payloads are stored server-side beyond transient buffers; failures are logged and the CLI continues immediately.

Note: Official RunMat binaries bake the ingestion key into the executable at build time (via the `RUNMAT_TELEMETRY_KEY` compile-time env var). If you build from source and want to talk to the hosted collector, set the same env var before invoking `cargo build`; otherwise the worker will return `401 unauthorized`.

### Delivery guarantees

The CLI keeps telemetry off the hot path via a background worker. By default it waits up to 50 ms for the `runtime_started` event to flush before exiting so extremely short scripts still count. This may add some wall time for sub-50 ms programs but is unnoticeable for longer runs. If you need faster exit times, set `RUNMAT_TELEMETRY_DRAIN=none` to prevent the worker from waiting for the `runtime_started` event to flush at all.

## How do I opt out of RunMat telemetry?

Use whichever method fits your workflow:

1. **Environment variable (one-off or CI):**

```bash
RUNMAT_NO_TELEMETRY=1 runmat script.m
# or
RUNMAT_TELEMETRY=0 runmat script.m
```

2. **Persisted setting:** create or edit `~/.runmat/config.toml` and set `telemetry.enabled = false`.
3. **Installer-only:** export the same variables before running `install.sh` or `install.ps1`.
4. **Delete identifier:** remove `~/.runmat/telemetry_id` after running with telemetry disabled; a new id will only be created if you re-enable the feature.
5. **Build-from-source:** if you compile RunMat yourself (e.g. `cargo install --path runmat`), telemetry stays off unless you export `RUNMAT_TELEMETRY_KEY=<ingestion key>` during the build. Keeping that env var unset is equivalent to a permanent opt-out for self-built binaries.

You can re-enable at any time by clearing the env var or setting `telemetry.enabled = true`.

### Source

- HTTP worker service: `infra/worker/`
- UDP forwarder service: `infra/udp-forwarder/`
- Terraform (DNS, Cloud Run, load balancer): `infra/main.tf`

---

Questions or feedback? Open an issue or reach us at team@dystr.com. We review this document regularly to keep it accurate and easy to read.

New to MATLAB? See the primer: [What is MATLAB? The Language, The Runtime, and RunMat](/blog/what-is-matlab).
