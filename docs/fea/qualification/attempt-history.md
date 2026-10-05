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

Native task completion now exposes the scheduler's committed result instead of
the raw worker response. In particular, a late worker success cannot override
scope cancellation. The result and complete original history are stored together
once; `NativeProgramTask::try_attempt_history` exposes the retained history after
terminal completion. Invalid scheduler evidence fails closed.

## Qualification boundary

This foundation preserves original history and prevents incomplete resource claims.
Unsuccessful backend reports still lack worker accounting. Driver loss/fencing does
not prove that a process stopped or quantify its consumption. Reserved CPU/memory
time remains distinct from measured utilization or RSS.

Remaining work includes original unsuccessful-attempt interval/transport evidence,
cumulative attempt admission, end-to-end retry/cancellation witnesses, propagation
through study usage/receipts, process-exit/transport accounting, and every claimed
execution placement. The local host now feeds a monotonic Unix-anchored clock into
the existing generic driver's deadline/cancellation policy while workers run. This
uses a weak host lifetime and does not copy the scheduler inventory per tick.
Native process deadline and user-cancellation witnesses exercise this without an
analysis coordinator. This document does not qualify the pending accounting gates
or any physics formulation.
