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

- `runtime_session_start`: `session_id`, `cid`, `run_kind` (`script`, `repl`, `benchmark`, `kernel`), `os`, `arch`, CLI version, whether acceleration/JIT are enabled.
- `runtime_value`: everything above plus `duration_us`, `success`, stringified error class (never source code), JIT usage flag, execution counters, provider metadata (device name/vendor/backend only), and GPU telemetry (dispatch counts, wall time, bytes moved, cache hits/misses, fusion stats).

Example runtime payload:

```json
{
  "event_label": "runtime_value",
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

- Clients send UDP (best effort) to `udp.telemetry.runmat.org:7846` or HTTPS POSTs to `https://telemetry.runmat.org/ingest`.
- HTTPS traffic lands on a Cloud Run service (`infra/worker/`) that validates a shared `x-telemetry-key`, normalizes payloads, emits human-readable summaries (so PostHog’s “URL / Screen” column reads like `run=script • jit=on • gpu=off`), and forwards to PostHog (primary) and GA4 (optional).
- The UDP path flows through a lightweight Google Cloud UDP load balancer into a managed instance group running the forwarder container (`infra/udp-forwarder/`). Each forwarder replays datagrams to the Cloud Run endpoint asynchronously so the CLI never blocks.
- No payloads are stored server-side beyond transient buffers; failures are logged and the CLI continues immediately.
- Official RunMat binaries bake the ingestion key into the executable at build time (via the `RUNMAT_TELEMETRY_KEY` compile-time env var). If you build from source and want to talk to the hosted collector, set the same env var before invoking `cargo build`; otherwise the worker will return `401 unauthorized`.

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