## Install Telemetry (transparent and optional)

We collect a tiny amount of anonymous telemetry during installation to answer basic questions:

- How many installs started/completed?
- Do we have regressions causing installer failures by OS/arch?

This helps us prioritize fixes and track the health of releases. We designed it to be anonymous, minimal, and easy to opt out of.

### What is sent

The install scripts emit at most one of each of these events per run:

- `install_start`
- `install_complete`
- `install_failed`

Alongside the event name, we send the following keys:

- `os`: operating system (e.g., `windows`, `linux`, `darwin`)
- `arch`: architecture (e.g., `x86_64`, `arm64`)
- `platform`: specific release target (e.g., `windows-x86_64`, `macos-aarch64`)
- `release`: release tag being installed (e.g., `v0.1.2`), or `unknown` on fallback
- `method`: installer type (`powershell` or `shell`)
- `cid`: a random, anonymous client id (see below)

Example payload (PowerShell):

```json
{
  "event_label": "install_start",
  "os": "windows",
  "arch": "AMD64",
  "platform": "windows-x86_64",
  "release": "v0.1.2",
  "method": "powershell",
  "cid": "9b7f0ee2-0a5a-4bda-8f1a-3a9d53f1b3e7"
}
```

### Anonymous client id

Both installers generate a random GUID the first time they run and save it to a local file:

- Windows: `%USERPROFILE%/.runmat/telemetry_id`
- Linux/macOS: `$HOME/.runmat/telemetry_id`

This id lets us count starts/completions without using personal or device identifiers. You can delete the file at any time; a new random id will be generated if telemetry is enabled later.

### Where telemetry goes

The installers post to `https://runmat.org/api/telemetry` (source code for that handler is [here](https://github.com/runmat-org/runmat/blob/main/website/app/api/telemetry/route.ts)). That endpoint:

- Forwards the event to Google Analytics (Measurement API) with the anonymous `cid`
- Stores nothing server‑side beyond standard, short‑lived service logs

If forwarding fails, the installer continues and your installation isn’t affected.

### How to opt out

Set one of these environment variables before running the installer to disable telemetry:

- `RUNMAT_NO_TELEMETRY=1`
- `RUNMAT_TELEMETRY=0`

Alternatively, you can remove the local `telemetry_id` file afterwards (see paths above). Future runs will remain opted out while those environment variables are set.

### Scope

Telemetry is emitted only by the installers. The runtime (`runmat` CLI) does not send telemetry.

### Source

- Windows installer: `website/public/install.ps1`
- Linux/macOS installer: `website/public/install.sh`
- API proxy: `website/app/api/telemetry/route.ts`

We welcome feedback. If you have questions or concerns, please open an issue or contact the maintainers.