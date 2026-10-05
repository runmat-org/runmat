# Shared original attempt history

The portable execution runner owns attempt identity, ordinal, driver fence,
immutable task request, scheduler disposition, and terminal evidence. This is
shared execution behavior; physics coordinators do not implement retry ledgers.

Snapshot schema 2 retains the terminal disposition separately from `AttemptState`:
the existing `Completed` state also represents execution failure. Successful worker
accounting is retained on the original attempt and must equal committed accounting.
Cold validation checks request/allocation identities, terminal-state consistency,
driver-clock bounds, original commit evidence, and contiguous attempt ordinals.
Old snapshots cannot supply missing original evidence and are rejected.

`task_attempt_history` returns every original attempt in ordinal order, including
active, rejected, failed, cancelled, and lost attempts. `task_resource_aggregate`
requires terminal accounting for every attempt. It rejects a successful retry
whose earlier attempt has unknown consumption. Planned reservations and a logical
driver closure time are not used to fabricate worker accounting.

Terminal records also retain original reported failure/loss diagnostics. Provider
observations bind the attempt/worker, invocation interval, actual local IPC reads/
writes, and root process exit status. These observations are distinct from complete
artifact accounting. A successful report with a nonzero exit or a different accounting
interval is rejected; cold validation enforces the same consistency. Missing
observations remain explicit for providers that cannot establish them.

Native task completion now exposes the scheduler's committed result instead of
the raw worker response. In particular, a late worker success cannot override
scope cancellation. The result and complete original history are stored together
once; `NativeProgramTask::try_attempt_history` exposes the retained history after
terminal completion. Invalid scheduler evidence fails closed.

## Cumulative requested retry work

Admission and cold restore derive a count-only cumulative reservation from the
shared retry policy and resource-accounting owner. Zero-attempt policies and
unrepresentable CPU/memory/artifact/transport projections reject before launch.
CPU rounding is applied per attempt before multiplication. Historical attempt
counts cannot exceed the original admitted policy. This projection bounds requested
work; it is not measured utilization, overlapping lost-process residency or proof
that cancellation completes within the requested interval.

## Native exchange and cleanup

The native worker now emits the same typed terminal envelope as domain workers;
the host accepts one message schema instead of probing a second raw response format.
Actual pipe metering includes framing, progress, partial reads and failed writes.
Counter overflow marks byte evidence unavailable. Local pipe bytes are not network
egress/relay bytes.

Every path after spawn attempts explicit cleanup and root-exit observation. Worker
success is rejected if the root exits unsuccessfully; wait and cleanup failures are
preserved instead of ignored. Cancellation remains active while waiting after a
terminal response. Positive native process identities are checked before signalling;
zero and out-of-range values never select a substituted process group.

## Native checkpoint persistence and delivery

The shared filesystem provider owns atomic replacement: a uniquely staged complete
payload is flushed before rename, and directory-flush failure after rename returns
visible-but-durability-indeterminate evidence with the original diagnostic. It is
never reported as an unpublished write. Unsupported providers fail explicitly;
ordering and fencing remain the storage consumer's responsibility.

Native snapshot capture and publication hold the existing driver state mutex. This
prevents an older snapshot from overwriting a newer terminal state. Snapshot content
is validated before replacement; the native consumer also flushes namespace ancestry.
Submission rollback occurs only after a write known not to have published. Cancellation
records persistence evidence before dispatching its existing shared cancellation actions.

Terminal result delivery follows snapshot persistence. `try_result` requires durable
checkpoint evidence; unavailable or indeterminate persistence returns a delivery error.
`try_scheduler_result` and original attempt history remain inspection evidence for the
original execution verdict, and do not authorize result publication. Task and session
checkpoint getters expose typed durability diagnostics. A storage error never rewrites
a successful physical execution as a failed execution or fabricates a retry.

Verification targets atomic-reader, original-snapshot, typed uncertainty and actual
native storage-fault cases; the evidence ledger records exact verdicts. These checks do not prove every crash/restart,
remote-provider, filesystem/hardware or resource envelope. Session retention remains
owned by the existing session-store lifetime.

## Qualification boundary

This foundation preserves original history and prevents incomplete resource claims.
Native unsuccessful attempts now carry observed interval/pipe/root-exit evidence,
while complete unsuccessful artifact/retained-output accounting remains absent.
Driver loss/fencing does
not prove that a process stopped or quantify its consumption. Reserved CPU/memory
time remains distinct from measured utilization or RSS.

Remaining work includes complete unsuccessful artifact/retention accounting,
reservation/measurement reconciliation and overruns, broader loss/retry evidence,
propagation through study usage/receipts, and every claimed
execution placement. The local host now feeds a monotonic Unix-anchored clock into
the existing generic driver's deadline/cancellation policy while workers run. This
uses a weak host lifetime and does not copy the scheduler inventory per tick.
Native process deadline and user-cancellation witnesses exercise this without an
analysis coordinator. This document does not qualify the pending accounting gates
or any physics formulation.
