# runmat-execution-runner

`runmat-execution-runner` is RunMat's portable, deterministic fine-grained
execution driver. It owns the task graph, resource scheduling, pool and worker
state, attempts, result fencing, cancellation, deadlines, retry policy,
event sequencing, and recovery snapshots.

The driver is an actor: callers enqueue ordered commands, the actor is the only
writer of scheduler state, and provider backends receive immutable attempt
requests and return reports. Backends never mutate driver state.

This crate has no native process, filesystem, network, Core, MATLAB `Value`,
test, Desktop, Server, tenancy, billing, or platform dependency. Native,
browser, and remote hosts implement its provider ports.

Snapshot schema 2 retains each attempt's original terminal disposition, logical
driver closure time, and available worker accounting. `task_attempt_history`
returns the complete ordinal history; `task_resource_aggregate` rejects active or
unaccounted attempts instead of reporting only the winning retry. A lost-worker
closure does not prove process cessation or resource consumption. Schema 1 cannot
be restored as complete terminal evidence. See the
[qualification boundary](../../docs/simulation/qualification/attempt-history.md).
