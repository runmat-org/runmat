# runmat-test-runner-execution

This crate is the narrow dependency-direction bridge between RunMat's test coordinator and its portable execution scheduler.

`runmat-test` and `runmat-test-runner` remain authoritative for discovery, fixture lifecycle, retries, cancellation policy, events, results, artifacts, reporters, and coverage. `runmat-execution-runner` remains authoritative for execution scopes, pools, resource admission, attempts, worker fencing, and result commit. `ExecutionWorkerBackend` maps one exact test attempt onto that scheduler without teaching either domain the other's semantics.

The same adapter wraps native process/session backends and browser Web Worker backends. `TestAttemptWorkload` is the encrypted remote payload: it binds the complete immutable plan and frozen source snapshot to the exact `ProgramRevision`, selected `TestId`, and attempt number. Remote results return the canonical `WorkerExecution`, so ordering, reporting, artifacts, and coverage are identical to local execution.
