# Original fluid temporal checkpoints and native block admission

## Authority

`FluidTransientCheckpoint` binds the complete authored workload, exact mesh,
compiled Cell-role property identity/audit, transient formulation and original
fresh initial state. It retains every original interval record, ordered block
ranges, cumulative conservative work charges and output ordinals. The terminal
state is borrowed from the last original record, including cell velocity,
pressure reference, pressure fluctuation and owner-oriented face flux. Neither
absolute pressure nor projected/nodal output is used to reconstruct restart flux.

Canonical decode returns claims. Source-bound recovery recompiles the original
source and initial conditions, rederives each block's inventory and work charge,
and evaluates every original backward-Euler interval. It does not solve, repair,
interpolate, normalize pressure or authenticate physics using hashes alone.
Corrupt scope, reordered/missing records, fields, charges and ordinals reject.
The checkpoint is immutable: successful advancement returns a new checkpoint.
Completed blocks share immutable original history storage in process; advancing
retains the original arrays without deep-copying all prior fields. Cold decoding
creates independently owned storage. Wire values and schema remain unchanged.

## Resource and execution boundary

A complete actual-schema profile reserves the maximum ledger inventory (one
single-interval block per authored interval), original arrays, metadata and
transport. Imported predecessor and output can coexist. The envelope is fixed
across splits so numerical identity does not depend on chunking/block partition.
Scope-initial copying has its own admission. Recovery workspace covers the
largest retained block, not only the next requested block.

Each successful continuation charges a conservative new-block solve envelope
plus the envelopes of prior blocks whose original histories it revalidates.
Cumulative charges are rederived before execution. They are not measured scalar
operation counts, wall time, RSS or a scheduler ledger for failed/retried attempts.

The distinct `FluidTransientBlock` resource phase uses shared host/transport
accounting, original source decode profiles and checkpoint output bounds. Native
workers authenticate immutable closures before source decoding and rederive the
exact source, checkpoint, CPU phase and output policy before advancing. Sealed
prepared resource facts avoid repeated history replay during transport planning;
worker preparation reuses its validated physical state through execution.
No second scheduler, property compiler, fluid integrator or private sparse solver
is introduced. Native output is an original checkpoint artifact, not a complete
study publication or production accuracy attestation.

## Evidence and remaining gates

Independent uniform acceleration/affine pressure references accompany full versus
split original-history comparison and source-bound cold corruption rejection.
The native witness runs separate child producer processes, waits for their terminal
exit, and resumes from canonical bytes in new processes. Uniform translation and
original history/chunk-policy equivalence exercise the actual native kernel's
immutable invocation and resource path. This is native-kernel process evidence;
it does not establish task-pool placement, retry fencing or remote execution.

Remaining work includes shared scheduler attempt accounting and cancellation during
execution, durable checkpoint/artifact import through fenced output sets, temporal
receipts, original observations/field histories, mandatory space/time accuracy and
atomic semantic publication. Selected activation, remaining shared boundary/source
histories, full typed cell/face state transitions, independent vortex/cavity/backstep,
conservative CHT/FSI, measured resources and all claimed backend/release gates remain.
Original scientific failures and the default-control continuity failure remain open.
No stage, production, accelerator, distributed or whole-release promotion.
