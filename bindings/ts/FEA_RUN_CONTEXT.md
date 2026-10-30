# Local browser FEA execution

Local browser mesh/solve needs no account, Server connection, core-hour lease,
or JavaScript billing adapter. `setFeaRunAdmission`, the admission interfaces,
and their feature sentinel have been removed. Deploy SDK and WASM together;
hosts must remove billing-adapter installation rather than retain no-op shims.

Checked-source planning, source revision checks, canonical study validation,
memory/scratch limits, runtime-owned preparation and scheduler execution remain
unchanged. A local Rust `FeaRunScope` retains allocated usage and nested-run
identity without acquiring or reporting a lease.

## Owned worker handoff

The parent passes a structured-cloneable `LocalFeaWorkerContext` to:

```ts
__runmatOcctExecute(json, context?)
__runmatExecutePreparedFeaStudy(json, memoryBytes, context)
```

The geometry profile query has no context because it does not mesh. For work,
the context contains the local run request, canonical authored study identity,
serial allocation (1000 millicores), and `preparedStudyIdentity`, the canonical
SHA-256 digest of the exact job JSON. There is no permitId or callback object.
The host binds dispatch/result delivery to its active owned worker and parent
run, checks the exact dispatched payload, and propagates cancellation/failure.
Context is not an authorization credential and cannot authorize cluster work.

The numerical worker forwards that context directly:

```ts
await executePreparedFeaStudy(json, memoryBytes, wasmModule, context);
```

Rust rejects mismatched payload/study identity, non-local site, non-serial
allocation and a meshing-only request before numerical execution. The native
local scope validates run identity and allocation and the prepared study's
existing execution path validates program/resources. A standalone local worker
can omit context; Rust creates a fresh local run identity with no billing gate.
The host still owns isolation, cancellation and worker cleanup; removing local
billing does not remove these responsibilities.
