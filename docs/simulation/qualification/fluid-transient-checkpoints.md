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
study publication or accuracy attestation.

## Evidence and remaining gates

Independent uniform acceleration/affine pressure references accompany full versus
split original-history comparison and source-bound cold corruption rejection.
The native witness runs separate child producer processes, waits for their terminal
exit, and resumes from canonical bytes in new processes. Uniform translation and
original history/chunk-policy equivalence exercise the actual native kernel's
immutable invocation and resource path. This is native-kernel process evidence;
it does not establish task-pool placement, retry fencing or remote execution.

## Scheduled execution and fenced cold import

The native block API submits the exact prepared closure and resource plan through
`AnalysisTaskPool`. It returns the original checkpoint, immutable host/response,
complete portable task evidence and usage across every original attempt. Shared
phase accounting verifies
allocation/capabilities and observed output bytes. Workers and cold consumers use
one source/admission owner. Cold import validates the exact closed output set,
source/resource plan, admitted payload/transport inventory, checkpoint identity,
original predecessor prefix and requested range, then rechecks the original equations.
Cold verification reuses the admitted original source instead of recompiling
properties and initial fields. It never reruns equilibrium or fills lost fields.

Actual scheduled process conformance executes uninterrupted and split accelerating
flow in three separate native sessions, retaining only immutable cold sources after
termination. Every original interval is compared exactly across partition/chunking.
Independent velocity and affine pressure references retain their original tolerances.
A 10^12 Pa reference accompanies nonzero original fluctuations; the witness confirms
that reported absolute pressure cannot reproduce the original fluctuation bits.
Foreign interval outputs, excess output inventory and altered allocation/byte
accounting reject. This is local scheduled execution and successful-attempt evidence,
not measured unsuccessful artifact/retention consumption, remote execution or a
completed study lifecycle.

The witness exposed a shared transport defect: native placement substituted its
own chunk policy for the policy used in task admission. The versioned host-v2
invocation now carries an explicit validated chunk/inventory policy. Every analysis
task preparation supplies it; worker settings are ceilings and cannot silently
replace it. The descriptor/program identities bind that policy. Missing/invalid
policy or old invocation versions reject, rather than falling back. Shared serial
tests exercise cold policy retention and rejection before physical input reads.
Output-set roots also reject their schema byte bound before any cache read.

Portable evidence retains the admitted task, ordinal history, scheduler commit and
observed driver clock/fence. Native completion freezes it alongside checkpoint delivery
status; storage failure cannot rewrite execution success. Cold validation checks
internal lineage, not trusted worker authorization. Local and remote analysis adapters
carry this evidence and preserve original terminal diagnostics. They use shared-driver
per-attempt/absolute deadlines rather than applying one attempt's wall limit to an
entire queued/retried wave. Their separate cleanup bound limits terminal collection;
a lost producer's logical closure does not prove process cessation.

The public fluid block result retains that carrier. Its usage graph expands every
reported original attempt against the independently admitted plan and uses the actual
host wall interval. Missing earlier accounting blocks complete usage; neither the
winner's accounting nor a reservation can fill the gap. Broader study coordinators
still consume terminal-only accounting and require the same complete-history cutover.

Remaining work includes cumulative failed/retried scheduler accounting, cancellation
qualification, temporal receipts, original observations/field histories, mandatory
space/time accuracy and atomic semantic publication. Selected activation, remaining
shared boundary/source histories, full typed cell/face state transitions, independent
vortex/cavity/backstep, conservative CHT/FSI, measured resources and all claimed
backend/release gates remain. Original scientific failures and the default-control
continuity failure remain open. No stage, qualification, accelerator, distributed or
whole-release promotion.
