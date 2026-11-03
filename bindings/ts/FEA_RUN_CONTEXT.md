# Browser FEA admission

FEA meshing and solving require sign-in, a Server connection and an available
core-hour allowance. This applies to local browser execution as well as cluster
execution. Ordinary local scripts, source checks and planning make no FEA lease
request.

Deploy the SDK, core WASM and OCCT sidecar from the same package. Install the
admission hook before running FEA. The page owns Server authentication; tokens
stay out of numerical-worker payloads.

## Install the host adapter

The host implements `FeaRunAdmission` with four methods:

| Method | Responsibility |
| --- | --- |
| `start(request)` | Acquire the original run's lease before any meshing or solving. |
| `heartbeat(leaseId, usage)` | Report cumulative CPU core-milliseconds and wall milliseconds; return the renewed hold and expiry. |
| `settle(leaseId, outcome, usage)` | Settle the original lease after owned work has retired. |
| `stopOwnedWork(request, reason)` | Await termination or normal retirement of this request's numerical workers. |

Create the supervisor on the page/UI thread so that its timers remain runnable
while numerical workers are busy:

```ts
const hook = createFeaRunAdmissionHook(
  admission,
  () => session.cancelExecution(),
  (runId, lease) => session.renewFeaRunLease(runId, lease),
);
session.setFeaRunAdmission(hook);
```

If the runtime session lives in a coordinator worker, proxy these calls through
the application's existing RPC. Keep the hook and its lease cache on the UI
thread. A raw `session.setFeaRunAdmission(admission)` wires the supervisor for
a session on the UI thread. Passing `null` denies new FEA roots.

The Server start body contains `runId`, `kind`, `allocatedCpuMillicores`,
`studyIdentity` and `client: "browser"`. Map those fields explicitly from the
runtime request; `site` and `estimatedWallSeconds` are runtime-only fields.
The Server lease contract is in `server/docs/FEA_CORE_HOURS.md` in the private
repository.

The supervisor enforces the current hold and expiry independently of heartbeat
completion. A partial hold permits only the usage it covers. Usage counts
allocated CPU cores multiplied by elapsed wall time; GPU time is excluded.

## Owned numerical workers

The parent registers the exact serialized payload with
`hook.delegate(request, payloadSha256)`. Its `LocalFeaWorkerContext` carries
the original request, payload identity, lease, root start time and delegation
generation. The host binds that context to its active owned worker.

```ts
const workerAdmission = {
  validateWorker: (context, phase) => uiRpc.validateWorker(context, phase),
};
await executePreparedFeaStudy(
  json, memoryBytes, wasmModule, context, workerAdmission,
);
```

The worker validates `start` before computation and `finish` before returning
result bytes. Both calls consult the parent's authentic lease state. Workers
make no second lease request. Missing adapters, payload mismatches, replayed
delegations and concurrent claims are rejected.

The parent retains the serial allocation until the worker has returned its
bytes and physically retired. For normal completion,
`stopOwnedWork(request, { reason: "completed" })` preserves the returned payload
and successful root. Cancellation, exhausted allowance, expiry or failed
renewal stops owned work before settlement. The parent checks its current
admission again before publishing results.

Use the same supervised boundary for the OCCT worker. Geometry profile queries
do not mesh and need no FEA lease. Mesh generation uses the canonical imported
source and meshing settings; its result returns to the parent for validation
and publication.
