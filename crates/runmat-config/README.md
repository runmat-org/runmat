# Runtime settings and host integration

`ConfigLoader::load()` is the native local adapter. `load_with_options` returns
`ResolvedRuntimeConfig { config, sources }`; `source("runtime.jit.enabled")`
reports the effective leaf's `ConfigSource`, including the originating file or
environment variable. Built-ins, user settings, project settings, explicit
`RUNMAT_CONFIG`, then environment are applied in that order. Tables merge
recursively; arrays replace in their entirety. Explicit JSON nulls replace
optional values. Unspecified fields do not acquire the provenance of another
field in their section. Provenance escapes literal dots/backslashes in key
segments with a backslash; use `source_segments(&["runtime", "logging", "otel",
"resource_attributes", "deployment.environment"])` for unambiguous dictionary
lookup without constructing the escaped path yourself.

Native user discovery first reads
`ApplicationSettingsDirectory::discover().settings_file()`
(`~/.config/runmat/settings-v1.json`), falling back to legacy `config.toml` then
`config.json` in the same directory only if that file is absent. A host supplies
its own location with `UserSettingsSource::File(path)`. A missing user file is
an empty layer; malformed/unreadable files are errors, not silent defaults.
Project discovery walks upward from `project_directory` using the existing
manifest filename order. `ConfigLoader::load_from_file` deliberately remains a
single-document parser; it does not inspect user preferences or environment.
CLI hosts can set `RUNMAT_USER_SETTINGS` to an explicit settings path (including
a missing isolated JSON file for offline tests), without changing `HOME`.
`ConfigLoadOptions::explicit_file` is a host-selected explicit file such as
`--config`; it layers above `RUNMAT_CONFIG` and below individual env overrides.

Filesystem-free browser hosts pass `UserSettingsSource::Document { source,
format: RunmatConfigFormat::Json }` and, when present, `project_document`.
`ConfigLoadOptions::default()` has no project discovery and does not read process
environment. Native callers use `ConfigLoadOptions::native()` to opt into both.
`UserSettingsSource::Disabled` opts out of user settings explicitly.

Cluster execution must use `ConfigLoadOptions::cluster_worker(project_root)`
or `ConfigurationPolicy::ClusterWorker` with a supplied project document. This
policy applies only built-ins and the bundled project document: it ignores
submitter/worker user settings, `RUNMAT_CONFIG`, and all ambient environment,
even if `environment` was set. Submitters must not serialize their resolved
local configuration as project policy. Cluster task payloads carry neither
user nor project collector settings or header secrets. Workers export to an
organization/admin process collector, if explicitly configured, or nowhere;
that startup authority never mutates the submitted document.

The transport captures `ClusterRuntimePolicy` schema 1: execution, language,
JIT, memory, acceleration, integrations, plotting and simulation settings only.
Logging and product telemetry are absent from this wire type. Native Direct
cluster bundles use schema 6 and bind the canonical policy object's descriptor
into the bundle identity. Remote selected Graph Programs use Graph schema 13
with their canonical project policy object included in authenticated Program
bytes. Native workers advertise `graph_project_runtime_policy_v13`; older
workers cannot be selected for that request. Graph child processes read the
policy from verified staged Program bytes, with their original scheduler
reservation carried in the typed launch reference. Worker GC defaults are
clamped to admitted memory/CPU ceilings; explicitly oversized GC requests fail.
Project preferences never grant devices or enlarge resource assignments.
Cluster session preferences/userpath use isolated memory, not the worker's home.

CLI job/tests/FEA/meshing cluster producers capture project policy before code
or domain-program issuance. The browser/SDK `prepareClusterProgram(sourcePath,
functionName?)` freezes the VFS project, resolves only its runtime document,
compiles under that language policy and returns `{ bundleArchive, descriptor }`
for the host's existing encryption/upload flow. It does not submit a run or
handle account credentials. It supports portable scripts/functions; building
native Python/Java/MEX artifacts requires a native producer.

For selected Graphs, `prepareClusterGraph(programJson, projectDocument)` or
Core's `cluster_settings::prepare_graph_for_cluster` attaches real project
policy before encryption/assignment. All selected code must already have the
same language and foreign authority as that project; otherwise the producer
reports that it must recompile for the cluster target. Harmless local GC/JIT
defaults are not treated as embedded semantic authority. Browser workers do
not advertise/execute native-cluster Graph V13; they can prepare/upload it for
native workers. Generic local Graph V10–V12 requests remain policy-free.

## Editing user settings

Use `UserSettingsDocument::load(path)` or `parse(json, Json)`, then
`edit(|config| config.jit.enabled = false)` and `save(path)`. Only changed fields
are written; full resolved configurations must never be saved as user defaults.
`set(&["jit", "enabled"], true)` writes an explicit override even when equal to
the built-in default. `reset(&["jit", "enabled"])` removes that override.
Segment arrays allow literal dotted resource-attribute keys. Browser hosts
persist `source()` through their existing preference store instead of `save`.

Unknown user fields at all levels are retained and ignored by the current
runtime; known values are still validated against the typed schema. Unknown
project/explicit runtime fields remain errors (to catch misspelled policy).
TOML edits preserve comments, inline tables, and untouched formatting; JSON
edits preserve values/unknown keys but normalize whitespace and key ordering.
`save` shares the preference service's application-document lock, atomically
replaces the file, checks original disk contents, and rejects stale editors. Null
overrides of non-null defaults require JSON, since TOML cannot represent null.
Unchanged default values are never materialized by `edit`.

## User-facing runtime fields

Acceleration restrictions are immutable per-invocation context, including nested
calls, asynchronous polls and parallel children; they do not replace the global
provider. `enabled=false` / `provider=inprocess` prevent new device computation.
Existing GPU values still support explicit gather through a source-bound numeric
readback capability: exact owner, live allocation and assigned device remain
mandatory, and transfer-only copy/map access grants no compute permission.
Retirement continues through the allocation's captured owner, independently of
the current compute preference.
`auto_offload.enabled=false` suppresses automatic uploads/fusion, but does not
forbid explicitly requested `gpuArray` operations on an admitted device.
`allow_inprocess_fallback=false` forbids automatic GPU-to-host retry and optional
provider-hook CPU implementations (including CPU compute followed by re-upload),
not ordinary CPU computation or an explicit `gather`. Unsupported-provider
errors retain their original cause when fallback is forbidden. FEA GPU requests
use strict accelerator admission under this policy; authored CPU solves stay legal.

Cluster provider preferences resolve only against registered providers matching
each exact authenticated device lease. Fixed worker inventories currently do not
attest WGPU power class or fallback-adapter identity: GPU attempts requesting a
non-`auto` power preference or `wgpu_force_fallback_adapter=true` are refused
before execution with an actionable error. They are never inferred from vendor
names. CPU-only attempts remain legal under default WGPU settings. Zero-worker
GPU Direct execution without an authenticated exact driver device assignment is
explicitly refused; requested resources alone are not device authority.

All names below are relative to the canonical document's `[runtime]` table.
The same schema and setters apply to user and project documents. Optional fields
may be omitted to inherit; `reset` removes only the selected user override.

| UI section | Fields and accepted values |
| --- | --- |
| Execution | `callstack_limit` (frame count), `error_namespace` (string), `verbose` (boolean); `jit.enabled` (boolean), `jit.threshold` (integer), `jit.optimization_level` (`none`, `size`, `speed`, `aggressive`) |
| Memory | `gc.preset` (`low-latency`, `high-throughput`, `low-memory`, `debug`), `gc.young_size_mb`, `gc.threads` (optional integer overrides); `gc.collect_stats` (boolean) |
| Acceleration | `accelerate.enabled`; `accelerate.provider` (`auto`, `wgpu`, `inprocess`); `accelerate.wgpu_power_preference` (`auto`, `high-performance`, `low-power`); `accelerate.allow_inprocess_fallback`; `accelerate.auto_offload.enabled`, `.calibrate`, `.log_level` (`off`, `info`, `trace`) |
| Language | `language.compat` (`runmat`, `matlab`, `strict`) |
| Simulation | `fea.artifact_store` (`filesystem`, `in_memory`), `.artifact_root`, `.study_artifact_root`, `.artifact_max_runs`, `.artifact_max_runs_per_kind` (optional paths/counts; zero retention count means unlimited) |
| Python | `foreign.python.executable`, `.version`, `.minimum_version` (default `3.9`), `.maximum_version`, `.execution_mode` (`in_process`, `out_of_process`), `.module_paths` (replacement array); exact captured policy: `.exact_version` (`major`, `minor`, `patch`), `.required_abi_tag`, `.required_platform_tag`, `.artifact_identities`, `.installed_module_paths` (capture-owner certificate, not an ordinary user switch) |
| Java | `foreign.java.home`, `.minimum_version` (default `8`, nonzero), `.maximum_version`, `.classpath` (replacement path array), `.options` (replacement string array) |
| Native/MEX | `foreign.native.isolation` (`process`, `in_process`), `.timeout_ms` (optional positive integer) or `.timeout` (`secs`, `nanos`, nonzero total duration; mutually exclusive with `timeout_ms`); `foreign.mex.unmanifested` (`isolate`, `deny`), `.timeout_ms` (optional positive integer). Native/MEX library paths are supplied by package/extension manifests, not a global unvalidated search-path setting. |
| Logs | `logging.level` (`error`, `warn`, `info`, `debug`, `trace`), `.debug` (boolean), `.file` (optional path) |
| Collector export | `logging.otel.enabled` (default false), `.endpoint` (base URL; default `http://localhost:4318`), `.protocol` (`http/protobuf`, `grpc`), `.signals` (replacement array of `traces`, `logs`, `metrics`; default traces and logs), `.headers_from_env` (optional environment variable *name*, default `RUNMAT_OTEL_HEADERS`), `.service_name` (default `runmat`), `.resource_attributes` (string dictionary) |

These are configuration contracts, not promises that every host has every
capability. Native Core sessions consume JIT threshold N as generic N,
specialized 2N and loop 4N; aggressive mode eagerly specializes at N. The
backend maps none → Cranelift `none`, size → actual `speed_and_size`, and
speed/aggressive → `speed`. Aggressive is a RunMat tiering policy, not a
separate claimed Cranelift optimizer. Reconfiguration creates a new compiler
cache generation while retaining active executor owners and installed project
authority. Browser execution has no native JIT; unsupported native integration
or filesystem FEA-storage settings report a capability error instead of silently
pretending to apply them.
Python does not expose a separate process timeout in the current typed schema.
Internal telemetry queue/drain controls, GUI capacity bounds and
`accelerate.wgpu_force_fallback_adapter` remain schema fields but need not be
shown in ordinary user settings. Plotting/GUI/export settings are defined by
`PlottingConfig`, `GuiConfig`, and `ExportConfig` and participate in the same
recursive layering and loss-preserving edits.

## Environment

Any runtime schema field can be overridden using
`RUNMAT_RUNTIME__<SECTION>__<FIELD>`, e.g.
`RUNMAT_RUNTIME__ACCELERATE__AUTO_OFFLOAD__ENABLED=false` or
`RUNMAT_RUNTIME__FOREIGN__JAVA__OPTIONS='["-Xmx2g"]'`. Values are JSON literals;
plain text is also accepted for strings. Arrays replace, tables merge, unknown
paths and invalid types fail validation. Generic overrides apply last in sorted
path order, with child paths after their parent tables.

Short names are also supported: `RUNMAT_CALLSTACK_LIMIT`, `RUNMAT_VERBOSE`,
`RUNMAT_JIT_ENABLED`, `RUNMAT_JIT_THRESHOLD`, `RUNMAT_JIT_OPTIMIZATION_LEVEL`,
`RUNMAT_GC_PRESET`, `RUNMAT_GC_YOUNG_SIZE_MB`, `RUNMAT_GC_THREADS`,
`RUNMAT_ACCELERATE_ENABLED`, `RUNMAT_ACCELERATE_PROVIDER`,
`RUNMAT_LANGUAGE_COMPAT`, `RUNMAT_LOG_LEVEL`, `RUNMAT_LOG_DEBUG`,
`RUNMAT_LOG_FILE`, `RUNMAT_OTEL_ENDPOINT`, and `RUNMAT_OTEL_ENABLED`.
Boolean shortcuts require `true` or `false`; count shortcuts require unsigned
integers. `RUNMAT_OTEL_ENDPOINT` and standard generic/per-signal OTel endpoint
variables also enable export; an explicit
`RUNMAT_OTEL_ENABLED=false` overrides this implicit enablement.

Common `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_PROTOCOL`,
`OTEL_SERVICE_NAME` and comma-separated `OTEL_RESOURCE_ATTRIBUTES` are honored
before RunMat-specific shortcuts. Per-signal OTel endpoints/protocols/timeouts
and all collector header secrets are resolved by the exporter at startup, never
copied to configuration. Hosts pass keychain secrets directly to startup.
Never store actual authorization headers in user/project documents.
`[runtime.logging.otel]` is independent of `[runtime.telemetry]`: changing one
never enables/disables the other.

## CLI and SDK adapters

`runmat config show --provenance` emits JSON effective configuration and exact
field sources, including explicit CLI arguments. `runmat config user show`,
`runmat config user set jit enabled --value false`, and
`runmat config user reset jit enabled` read/edit only the application user file;
`--file` selects an explicit host settings file. Inspection/editing never starts
the execution/GPU/export lanes.

SDK `runtimeSettings` accepts `HostRuntimeSettings` (`userSettingsJson`,
`project: { source, format }`, `policy: "local" | "clusterWorker"`). Explicit
legacy init options override only fields actually supplied; omitted SDK defaults
do not erase user/project settings. `resolveRuntimeSettings` returns effective
typed settings and sources without starting a session;
`editUserRuntimeSettings` returns loss-preserved JSON for the host's settings
store. Shared application settings keep durable preference-version validation:
settings-only documents hydrate empty preferences, but preference fields require
an explicit supported version; null typed fields and old/future versions fail.
