# Browser FEA admission

The host installs a `FeaRunAdmission` with
`session.setFeaRunAdmission(admission)`. Planning, checks and geometry previews do
not need a lease. Running a checked source asks `admit` once before its first
mesh. An unset adapter denies with `RM.FEA.SignInRequired`.

The adapter creates the Server core-hour lease with `clientSurface: "browser"`,
validates its signed token (signature, audience, expiry, lease/org/run identity
and allocation), and returns its `leaseId` as `permitId`. The SDK does not hold
credentials, implement sign-in UI or validate the Server token for the host.

Usage is cumulative allocated core-milliseconds and wall-milliseconds, not CPU
time. The browser serial allocation is 1000 millicores. The parent reports about
every 30 seconds and settles once for the whole mesh/solve run, including all
load and frequency cases. Completed and cancelled runs charge usage; failed
runs release the reservation. Dropping an unfinished run reports `failed`.
The host must journal heartbeat/settle before transport and retry offline
reports; reporting failures do not interrupt an admitted run.
Usage that cannot be represented as an exact JavaScript integer is never
silently rounded or clamped: it emits `RM.FEA.AccountingInvalid` to the console
and leaves the lease for host reconciliation instead of sending an inexact report.

## Owned worker delegation

The runtime passes a `FeaRunDelegation` to the owned host callbacks:

```ts
__runmatOcctExecute(json, delegation?)
__runmatExecutePreparedFeaStudy(json, memoryBytes, delegation)
```

The optional delegation is absent for the geometry `profile` query, which does
not mesh. For actual work it includes the admitted request, parent permitId and
`preparedStudyIdentity`, the SHA-256 digest of the exact job JSON.

A delegation record is context, not authorization. The host must bind it to an
active admitted parent run and the exact dispatched job before starting its
worker. Worker authorization can use an RPC to that host; functions cannot be
structured-cloned into a worker. The numerical worker calls:

```ts
await executePreparedFeaStudy(json, memoryBytes, wasmModule, {
  admission: workerHostAdmission,
  delegation,
});
```

`workerHostAdmission.admit` must authorize the existing parent lease and return
the same permitId, without creating another lease. It checks runId,
studyIdentity, allocation and exact job digest against the parent job. The
worker additionally checks the job digest, serial allocation and returned
permitId. Only the parent heartbeats and settles a delegated run. The host must
propagate cancellation/worker failure to the parent so it settles accordingly.
The exact-geometry worker needs equivalent host authorization before meshing.

A standalone numerical worker can omit `delegation` but must still supply an
admission adapter; then it owns its lease and reports usage. Omitted admission
fails closed. The SDK also refuses older numerical WASM assets that do not
expose `feaRunAdmissionSupported()`, so mixed assets cannot silently bypass the
new gate. Rebuild and deploy SDK and WASM assets together.
