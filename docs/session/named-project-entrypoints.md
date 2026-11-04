# Named project entrypoints

The browser SDK selects `[entrypoints.<name>]` from the enclosing `runmat.toml`
or `runmat.json`. Resolution belongs to `runmat-config`: targets are either a
declared path or a module/function pair. The captured manifest must match the
canonical manifest digest in the frozen project. Selected code is captured by
Core's existing primary-source authority; JavaScript does not resolve source
roots or synthesize a call expression.

An owning session exposes `listProjectEntrypoints(projectPath, context?)` and
`runProjectEntrypoint({ projectPath, name }, options)`. Each description contains
`name`, a typed `kind` (`script`, `typedFunction`, `feaStudy`), the declared
`target`, and separate `local`/`cluster` support records. Unsupported support
records contain a structured `reason.code` and `reason.message`. These describe
this browser producer, not another host or available scheduler capacity.
Invalid or unresolved targets reject with a structured diagnostic; they are not
reported as executable merely because a filename has a recognized suffix.

`context` accepts `userSettingsJson` and `inputManifest`. Run options accept the
same fields. The selected project's exact captured manifest supplies the project
layer; the initial typed user layer and explicit host overrides are retained,
not the previously selected project's merged settings. An explicit per-request
user JSON overrides that retained user document. Local compilation and execution
apply this one resolved policy. Listing uses temporary compiler sessions and
never changes the live session's project, configuration or FEA plan.

Startup-owned GPU, GC, plotting, logging or FEA storage changes require a host
initialized with the selected project settings. Listing reports local
unavailability and running rejects `project.entrypoint.hostReinitializationRequired`;
it never silently advertises those settings as applied.

Scripts run through the existing session `executeRequest` owner. They accept an
optional immutable input manifest, but no positional arguments or ordinal output
request other than zero. Typed functions select their original compiler function
identity and invoke the existing immutable executable owner. `argumentsJson` is
a `Uint8Array` containing UTF-8 JSON `Vec<runmat_execution::value::ValuePayload>`;
the canonical Rust codec validates the complete aggregate and retains argument
roots through invocation. `requestedOutputs` defaults to one. The typed function
result is UTF-8 JSON of the ordinal `Vec<ValuePayload>` in a `Uint8Array`. Byte
transport deliberately avoids rounding 64-bit integer or floating-point bit
patterns through JavaScript numbers; this is not the preview-value format.

Run results are discriminated records: `{kind: "script", result: ExecuteResult}`,
`{kind: "typedFunction", result: Uint8Array}`, or
`{kind: "feaStudy", result: FeaRunResult}`. Script/FEA input manifests retain their
existing immutable-provider admission. Function invocation retains session
ownership, cancellation and the canonical output codec. No replacement local
runtime or source interpreter is created to execute a named target.

`cancelExecution()` also covers asynchronous project/source discovery and
read-only FEA preparation. Cancellation stops waiting for those reads, rejects
with `RunMat:ExecutionCancelled`, and releases the session for the next request.
The filesystem provider may still finish its outstanding I/O; cancellation does
not detach a RunMat execution or publish a result from the cancelled preparation.

`prepareClusterProgram({ projectPath, name })` selects the same named target and
delegates to the existing project-only cluster producer. The legacy path plus
optional function-name form remains available. A named selection cannot also
supply an overriding function name. Cluster preparation does not inherit laptop
user settings or relabel a locally compiled product as a cluster product.
An optional `inputManifest` on that named selection applies before manifest,
target and source discovery. Cluster availability is compiled independently under
project-only policy; local/user compatibility is not a cluster-capability guess.
The actual producer and availability use canonical typed interop admission,
including Java, MEX, native FFI and Python requirements. This browser installs no
native foreign adapters or generic foreign host bridge.

All named paths capture provider and CWD once. Filesystem-owned per-poll scopes
retain immutable inputs without changing the process-global provider. Selection
and compilation use read attenuation; execution reuses Core's cached canonical
pinned provider and owned-output registry. Unpinned execution and FEA result
publication use the retained original effective authority. Scoped user `cd`
changes the invocation-local directory, not another session or the process CWD;
selected Runtime invocation rights and directory ownership remain authoritative.

## Browser FEA limitation

Named local FEA uses the existing checked-source browser plan/run path, including
geometry/assets, immutable inputs, exact checked revision, result-publication
capability and memory readiness. Listing shares that readiness owner without
installing a project or plan into the live session. The current browser route is
the existing serial structural-study subset; other checked studies report their
actual unsupported readiness instead of claiming general FEA support.

Browser FEA **cluster preparation is unavailable**. Both listing and the actual
named preparation entrypoint report
`project.entrypoint.browserFeaClusterUnavailable`. Preparation validates the
project-only policy and checked FEA source before that refusal. A serial browser
study is not serialized as a code bundle.

The source-defined native alternative is the OCCT-enabled CLI:
`runmat run <named-entrypoint> --cluster <cluster>`, subject to its existing
study selection, native geometry capability and authenticated cluster admission.
Its owners are `commands/fea/preparation.rs`, `run/remote_study.rs`, and native
`typed_fea/preparation.rs`. This documents a source route, not a new qualification
claim or a claim that a Desktop caller is wired. Moving the native portable
geometry/meshing/root producer into a WASM-compatible shared owner is separate
architectural work; importing NativeRunner into WASM is not a supported shortcut.
