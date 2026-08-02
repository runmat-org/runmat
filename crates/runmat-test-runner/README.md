# runmat-test-runner

Portable coordination and reporting for immutable RunMat test plans.

This crate owns scheduling, isolation selection, cancellation/timeout escalation,
crash classification, event sequencing, retries, and report fan-out over
host-supplied ports. It intentionally contains no process, signal, filesystem,
JavaScript Worker, DOM, or IndexedDB implementation.
