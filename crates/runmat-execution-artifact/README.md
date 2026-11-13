# runmat-execution-artifact

Canonical, filesystem-independent execution bundles and exact program artifacts.
The crate consumes an already frozen package graph; it does not resolve packages,
schedule work, execute programs, authorize tenants, or expose physical paths.

Schema-v6 bundles carry the existing frozen-project handoff with its storage paths rebound to canonical logical source objects, or an exact compiled code closure. A native or browser worker validates the graph, source catalog, program revision, object digests, callable inventory and project runtime policy before rebasing that same handoff onto a private host-owned materialization. Workers never resolve package locators, fetch Git/project/registry sources, or receive package credentials.

`BundleManifest.project_runtime_policy` names one optional digest-bound
`ProjectRuntimePolicy` object. It is optional only for generic local/noncluster
bundles: cluster producers/receivers must require the object and losslessly
capture/decode Config's typed `ClusterRuntimePolicy` before execution. There is
no worker-CWD or submitter effective-user-settings fallback.

The object has schema version 1, media type
`application/vnd.runmat.project-runtime-policy+json`, a digest-derived
`project-runtime-policy/<digest>` logical name and a 64 KiB maximum. Recursive
JSON object ordering is canonical; arrays preserve order. Typed decode must
re-encode to the exact original object, rejecting ignored fields or defaults
that would change captured authority. Config owns the complete execution schema;
the transport admits only its versioned object envelope, never a replacement
generic config. Collector/logging/product-telemetry/header settings are excluded;
worker OpenTelemetry remains administrator/process-owned. Portable environment
snapshots also reject OpenTelemetry environment overrides and headers.

The descriptor participates in the schema-v6 bundle identity and canonical
manifest archive. Exact object inventory rejects missing, extra, substituted or
duplicate policy objects. The reader checks descriptor size/media/name before
reading policy bytes. Selection from an admitted parent bundle retains its policy
object; fresh compiled-unit constructors require explicit attachment by their
producer. The schema-v6 manifest domain rejects schema-v5 bytes rather than
silently accepting absent policy authority. Archive framing itself is unchanged.
