# Thermomechanical numerical transport memory evidence

Internal qualification record moved from the Thermal stress user page. It records how thermal
source results, receipts and bridge transport bind identities and what memory each path was
measured to use. User documentation for thermal stress is in
[physics/thermomechanical.md](../physics/thermomechanical.md).

Thermal source-result identity binds producer-recorded numerical intent and shared recovered-field identities for every retained
temperature, gradient, flux and storage-rate-density sample. Fields are identified one at a time
from borrowed storage; the final identity envelope contains field identities and numerical
metadata, not all history values. Snapshot identifiers and physical times remain bound. Shared
owned-field construction uses the same validating identity function without hashing twice.

The result retains separate mandatory exact-workload and numerical-intent bindings. Numerical
intent uses the shared solver-model contract, selected step and thermal controls, excluding backend
and formulation-admission requests. The exact workload binding also retains those requests.
Matching only sequence/step and schedule was insufficient: a result could otherwise be relabeled
with a changed conductivity model. Compact thermal numerical receipts now check this binding,
the exact accepted schedule, reference temperature, backend request and residual target against
the retained result manifest. Tet4/Tet10 tests reject material substitutions even when the sequence
is unchanged. The receipt is accepted-solver evidence, not an independent accuracy report; mesh
binding and portable worker receipt routing still require the task integration path.

Per-solve global energy balances and the residual target are mandatory retained numerical metadata,
including in the partition manifest. Steady runs retain one balance; transient runs retain one per
advance, not for the initial-only sample. Validation recomputes the signed power residual
`storage + boundary_loss - applied_power + prescribed_boundary_power` and its normalization by
the largest retained physical transfer scale, without a dimensional floor. Scales retain nodal
L1 sums of storage, applied heat and prescribed reactions, and the integral of absolute net Robin
transfer, before global signed cancellation. Each balance also retains the gross heat entering
and leaving through prescribed-temperature nodes, so a body held between two fixed temperatures
reports its through-flow (for a bar, `k A (T_hot - T_cold) / L`) even though the net prescribed
reaction is zero; `runmat run` prints it as `heat through prescribed temperatures`. Exactly zero transfers have zero relative residual;
uniformly scaling all powers does not improve acceptance. All terms must be finite, each scale
must bound its signed power, the recorded
arithmetic must agree, and both linear and normalized energy residuals must meet the target.
Receipts bind the target to the authored workload and retain both maximum residuals. Diagnostic
severity/text is no longer acceptance authority. These checks do not independently establish
engineering accuracy or verify power terms against unread field histories.
The physical balance separates applied generation/flux from the algebraic Robin ambient forcing:
boundary loss is the net outward convection integral `h * (T - T_ambient)`, not the sum of the
conductivity matrix applied to absolute temperature. Internal conduction redistributes heat;
it is not external boundary power. Prescribed-temperature reaction power includes conductive
exchange across prescribed/free-node pairs, storage, applied heat and convection. Temperature
differences are evaluated before gradient products so uniform fields do not create fictitious
conductive power. These reaction terms are equation-based accounting, not an independent
verification of a fully prescribed thermal solution. A tiny load unresolved by the numerical
temperature solution must still fail conservation; absolute-temperature energy is not a substitute
normalization scale. Opposing prescribed heat flows retain their individual nodal-reaction scale
even when the net reaction vanishes. Quadratic-element signed source weights can make their nodal
L1 scale larger than the integrated applied power; this is explicit normalization, not an absolute
tolerance. Scales are required persisted metadata and participate in result identity; old claims
without these terms are not silently upgraded.
Steady solves use a constant temperature reference per connected body, selected from its prescribed
temperature or ambient boundary when available. The increment equation forms Robin differences
before multiplication and retains the original nullspace/anchor checks. Reconstruction restores
authoritative prescribed temperatures exactly; free-node increments are not clamped or overwritten.
Thermal solver stopping uses the tighter of the shared default and authored acceptance target;
the latter cannot relax the established numerical stopping policy. The shared sparse solver layer
propagates this tolerance through its direct-refinement,
CG and accelerator final-residual checks. Fixed-step transients reuse one Cholesky factorization
for every step whose effective operator is unchanged. No new successful
GPU execution envelope is inferred. Linear and global energy normalizations differ, so converged
linear solves must still pass the separate energy gate. Producer validation now runs before final
completion; its uniqueness-validation scratch remains outside complete producer-memory admission.

The accepted-intent-bound identity domain includes target and balances and intentionally changes thermal source-result identities; there is no legacy
identity fallback. Shared recovered-field canonical limits apply to each field, and negative-zero
sample times are rejected consistently. A thermal result may fit its monolithic serialization
limit yet exceed a shared per-field identity limit; publication fails closed in that case. This
does not expand the qualified large-field envelope. The canonical serializer still materializes
one field at a time.

Thermal result manifests now bind all four retained field inventories and their shared recovered-field
partition descriptors. Metadata validation and numerical identity are shared with the full retained
result; the manifest contains no observation or field-history values. Native state publication
decodes this manifest and only the terminal-temperature partitions. The artifact-cache importer can
publish the same state with all unrelated history blocks absent. It validates request lineage before
reading numerical blocks, checks the exact declared inventory, and bounds the selected closure's
aggregate bytes and object count before fetching temperature blocks. Selected partition corruption
fails closed. This partial import does not independently validate unread historical values.

Preparation still retains encoded partitions and output chunks, and the existing native prepared
input closure can retain all chunks. This is not streaming producer execution or a whole-process
memory bound. Portable thermal Solve/task/receipt routing and native partial input acquisition
remain open; the isolated state-publication path does not qualify whole-study thermal execution.

The shared canonical codec can report both conservative encoded size and materialization residency
from a single borrowed Serde traversal, without constructing encoded bytes or a value tree. Each
separately encoded field partition needs its own contract envelope. Thermal producer metadata and
overlapping output construction still need domain-owned sizing before portable worker admission;
a generic manifest schema limit is not a demonstrated useful thermal execution envelope.

Preassembly now reserves shared-codec materialization estimates for initial-state evidence and
the five producer-owned physical diagnostics. Initial evidence projection and realization share
the authored temperature-condition selector and one serialization schema. Projection borrows IDs;
the shared step-applicability check can still allocate a transient object-reference ID. Diagnostic
projection and reporting share one wire schema and message templates. Finite floats use roundtrip
17-significant-digit scientific formatting, bounded by 24 bytes; integer widths use `usize::MAX`
and the concrete digest formatter supplies its 71-byte representation. This traversal creates no
dummy numerical arrays or retained message strings. Actual initial evidence and physical messages
are checked against their projected encoded and resident estimates. These are size checks, not
independent semantic validation. Material-assignment conflict diagnostics now share their borrowed
selection, message templates, and wire schema with preassembly profiling; matching assignments
produce no diagnostic. Successful shared SPD solve diagnostics are bounded across CPU, admission
fallback, accelerator, sparse Cholesky, AMG, Jacobi and ILU0 alternatives. Thermal checks the last successful solve's list before
cloning it for output and reserves two lists for previous/current solve or final-report overlap.
Arbitrary provider failure strings are not part of this bound: thermal rejects nonconverged solves.
Successful backend evidence now has a producer-owned profile through its derived serializer, with
the optional preconditioner binding present. Preassembly reserves the scheduled solve count plus
one in-flight record, including inline vector slots. Accepted records are validated and moved out
of the current solve into a lazily reserved inventory; CPU-only execution allocates no inventory.
This removes duplicate retained evidence in the last solve. Fixed-width typed IDs can allocate
temporary strings during profiling. These are evidence-metadata bounds, not accelerator device
memory qualification. Producer result validation now reserves one entity-ID scratch vector sized
to the larger node/element inventory. The sequential uniqueness checks preserve arbitrary declared
order, sorting a fallibly allocated copy only when IDs are not already ascending. Thermal histories
reuse shared single-field validation and exact ordinal-derived identifiers, which establish
uniqueness without a history-sized set. The generic portable-field inventory validator still checks
arbitrary imported inventories separately; the thermal producer emits no additional `run.fields`.
This does not claim allocation-free validation: expected-ID formatting and rejection messages can
allocate. Manifest validation and overlapping transport construction remain open admission work.

Post-solve thermal preparation now profiles actual manifest metadata and hypothetical field
partition shapes through the same serializers used for retained fields. No dummy numerical arrays
are allocated. The profile counts each canonical contract, sums encoded-size bounds, and records
the largest individual materialization estimate. Oversized payload projections and inventories
that cannot hold at least one chunk per contract plus a root are rejected before final numerical
payload encoding. These conservative guards may reject an encoding that would happen to be smaller;
final transport validation still checks actual chunks and the root. Manifest construction has
already encoded field and partition identities before this guard runs.
This does not admit that construction, retained histories, simultaneous buffers, or pre-solve
runtime provenance. The largest-contract estimate is reported, not a process-memory limit.

Manifest descriptors and transport preparation now use the shared validated `RecoveredFieldView`
over immutable borrowed snapshots. Canonical extraction yields one owned partition at a time;
descriptor construction drops its numerical values before advancing, and transport appends each
encoded partition to the shared incremental chunk builder before dropping both temporary buffers.
Neither consumer clones a full numerical snapshot
or retains all extracted numerical partitions. Manifest identity reuses the field identities already
computed for its entries. Source, partition and result identities and canonical wire formats are
unchanged. The owned collecting API delegates to the same iterator for consumers needing all parts.

`cargo test -p runmat-analysis-fea --test field_partition_residency -- --nocapture` measures
traversal separately from initial full-field identity construction. For scalar fields with 4 and 48
canonical partitions (524,288 and 6,291,456 source bytes), additional peak Rust heap was 1,577,498 B
in both cases. This is bounded traversal evidence only: full-field identity encoding still
materializes a serde value tree and canonical buffer. These temporaries, retained output chunks,
and manifest validation still need complete resource admission.

Field-construction scratch is now a thermal preassembly component. The recovered-field identity
shape profiler uses the exact identity tuple and core host-field serializer with counted hypothetical
scalars, without allocating numeric arrays or a value tree. Thermal profiling uses the same snapshot
definitions as publication and the longest scheduled ordinal suffix. It reserves the larger of
full-field identity materialization and one partition's extracted values/indices plus canonical
materialization: those operations are sequential, while retained histories are reserved separately.
Tests cover real/complex scalar, vector and symmetric-tensor identity schemas, invalid profiles,
overflow, and Tet4/Tet10 solve rejection when this scratch is omitted from the resident ceiling.
This component alone does not admit retained manifest/transport inventory or their validation overlap;
it is not a whole-process memory qualification.
The isolated field-residency test also measures identity construction separately: 65,536 scalar
values used 3,149,451 B peak additional Rust heap against a 5,444,494 B profile; 786,432 values used
33,558,155 B against 64,557,966 B. These two measured cases support the profile's conservative
scratch estimate; native allocator overhead and unrelated process memory are not included.

The retained thermal sample/field/partition catalog now has a separate preassembly component.
Its shape projection uses the same artifact-entry serializer and actual partition-descriptor schema
as publication, repeating only borrowed templates rather than allocating descriptor arrays. It
includes the shared catalog materialization estimate, an explicitly sized partition-identity sort
buffer, and the result-identity vector. Manifest uniqueness validation uses a fallibly reserved
sorted copy, preserving the original catalog order; this buffer drops before identity encoding.
Construction also drops its temporary identity list before manifest validation, avoiding two live
copies. Tet4/Tet10 tests compare projected and actual catalogs and reject duplicate partition
identities and insufficient resident ceilings. This still excludes the manifest's other run/source
metadata and full transport-object/chunk lifetime composition; complete admission remains open.

Thermal source evidence now separately reserves canonical serialization materialization in addition
to retained Rust records and identifier strings. A borrowed history traversal uses the actual
spatial-evidence serializer, shared active-action selection, authored IDs and projection requirements,
and all scheduled samples (including the initial assembly). Static and every time-approach shape
are covered without allocating a dummy record history. Successful execution checks the actual
evidence count and profile against admission. Tet4/Tet10 volume/flux tests cover the complete
history and resident ceiling; metadata growth and missing records are rejected. This closes the
source-history serialization component, not the remaining full manifest wrapper/array metadata
or transport lifetime bound.

Complete canonical header scratch is now admitted separately from retained producer evidence.
The projection uses the actual manifest serializer with empty variable collections and the same
registry-issued formulation identity later used by the solver. Exact nonstructural admission is
shared by planning and execution, rather than reconstructed by the estimator. Typed repeated
node/element IDs, time points, residuals and energy records fill the inventory bound without dummy
arrays; existing initial-state, diagnostic, backend, source and catalog profiles are added with
checked arithmetic. Shared-codec estimates are positive linear sums plus fixed overhead, so this
composition conservatively double-counts wrappers rather than dropping their cost. Tet4/Tet10
steady and transient headers are compared against this bound, including source-bearing results
and exact resident-ceiling rejection. This is canonical materialization scratch, not a claim that
all transport lifetimes or whole-process memory are qualified.

Retained manifest metadata now has a separate preassembly reservation using actual Rust record
sizes and producer-owned inventory counts. This includes scalar inventories, nested admitted
identity vectors, source records and IDs, diagnostic records and strings, successful backend
records, and initial-condition IDs. Catalog storage remains separately reserved. Canonical encoded
byte bounds cover string payloads only; they are not treated as estimates of typed record storage.
Numerical-result identity now shares one tuple serializer between execution and shape profiling,
with a legacy-wire parity test. Identity validation and header encoding are sequential scratch
phases, so their maximum is reserved alongside the retained metadata and catalog.

The isolated `thermal_manifest_residency` test holds the source result live while constructing,
validating and encoding its manifest. Tet4/Tet10 fixtures with source evidence, long UTF8 IDs and
8/128 snapshots measured incremental requested-heap peaks of 248,195–3,568,584 bytes against
composed reservations of 1,718,680–17,716,280 bytes. Tests also cover exact resident-ceiling
acceptance/rejection and checked overflow. These are fixture measurements of Rust allocations,
not allocator/native/RSS qualification. Actual chunk-size-dependent transport inventories,
output accumulation and their overlapping lifetimes remain open.

Thermal output preparation now preflights the complete wire closure before constructing transport
chunks. The shared numerical-object profiler sums the chunk ceiling of each independently encoded
partition using the requested chunk size, then profiles the actual shared root serializer. Root
bytes and the root object count toward inventory limits; both root and chunks obey per-object
limits, and the root obeys its canonical schema limits. Tests cover partition remainders, overflow,
root limits, unchanged wire bytes, and exact projected-limit acceptance/rejection at two chunk sizes.
The projection is conservative and may reject a closure whose eventual encoding is smaller.
It avoids hypothetical chunk/descriptor vectors, but custom serializers may allocate temporary
strings. This is post-solve wire admission, not pre-solve transport or complete retained-heap
admission: builder uniqueness trees, artifact inventory validation and object-conversion overlap
still require explicit memory accounting.

Bulk numerical-manifest and shared artifact-inventory uniqueness validation now uses one fallibly
reserved vector of borrowed keys and in-place unstable sorting instead of tree nodes. The respective
owners expose checked requested-slot byte counts; transport projection includes the manifest's
validation component. Canonical ordering and accepted inventories are unchanged. Artifact validation
also preserves earlier error precedence; for a manifest with multiple simultaneous defects, the
reported defect may change. These components exclude error strings, allocator overhead and RSS.
The isolated manifest-validation test measures 8, 512 and 32,768 requested heap bytes for
1, 64 and 4,096 partitions on the tested 64-bit host, exactly matching the reserved reference slots.
Object-conversion overlap is included in the planned bridge projection described below.

`AnalysisPayloadBuilder` is the single incremental chunking authority; the bulk borrowed-slice
entrypoint delegates to it. Its constructor consumes an `AnalysisPayloadPlan` with unique ordered
identities and per-partition encoded bounds. The plan uses the actual chunk size and shared root
profile, and reports requested heap for its owned capacity, output-vector slots, accumulated payload
bytes, one partition's staging slots and validation references. Output vectors are reserved fallibly
before encoding. Each append checks the next planned identity and byte bound, then stages its chunks
fallibly before changing hashes or inventory. Missing, duplicate, unexpected or out-of-order entries
cannot produce a completed result. No incremental identity tree remains. Canonical chunks still restart
at each semantic partition boundary. Thermal preparation no longer retains an all-payload inventory
alongside those chunks. Chunks move into transport objects through the existing validated bridge.
Tests compare complete Tet4/Tet10 thermal prepared closures against collected-input preparation,
including two chunk sizes, and independently specify small partition/chunk digest expectations.

`cargo test -p runmat-analysis-fea --test incremental_chunk_residency -- --nocapture` includes
generated input buffers and retained output chunks. With 512 KiB partitions and 64 KiB chunks,
the test now compares peak requested heap against the checked plan reservation plus one generated
source partition, replacing a fixed metadata allowance. The plan exists before the measurement
baseline, so including its retained capacity in the tested bound is conservative. This is not full
thermal serialization, producer residency or an enforced caller heap ceiling. Pre-solve
whole-lifetime admission remains separate work.
The checked-plan implementation measured 8,923,712 B for 8,388,608 output bytes and 34,121,408 B
for 33,554,432 output bytes on the tested host; both fit that composed bound.

The execution bridge's `projected_chunked_output_resource_facts` composes the payload plan with
retained `LogicalObject` slots, logical-name/media strings, the root descriptor copy, root canonical
encoding/materialization and borrowed inventory-validation keys. Conversion reserves strings and
inventory vectors fallibly; chunk byte buffers move without copying. Canonical names and bytes
remain unchanged. `projected_transport_output_resource_facts` uses the same formula before
partition identities exist. The conversion component alone excludes source fields and their
canonical encoding, allocator overhead and native/process memory.

`cargo test -p runmat-analysis-execution --test object_conversion_residency -- --nocapture`
measures the public bulk preparation path above caller-owned input buffers. For 16 and 64 partitions
of 512 KiB, with 64 KiB chunks, peak requested growth was 8,560,281 and 34,237,545 B against
8,697,944 and 34,582,712 B reservations on the tested host. It also checks the root encoding and
legacy logical-name/media identity. These fixtures are not whole-process memory qualification.

The local `execute_thermal_solver_result_objects` bridge now calls `ThermalWorkload::prepare`
before assembly. This immutable preparation owns the existing admitted numerical facts and compiled
properties, borrowing the unchanged model and mesh. Its output projection combines the complete
header bound with full/trailing field block shapes at the longest sample-name ordinal, using the
actual chunk policy. It checks the thermal header byte ceiling, shared root limits, artifact count,
per-object and total bytes. The bridge then adds builder/conversion residency to the conservative
numerical/producer/field-encoding reservation and enforces the selected step's resident ceiling
before consuming preparation to execute. It does not rerun property compilation or introduce
transport policy into numerical identity. Summing lifetimes is deliberately conservative and may
reject cases that a tighter liveness analysis could admit.

Tet4/Tet10 steady and 12-sample transient tests verify analytic uniform temperature, unchanged
prepared results, and pre-solve bounds covering post-solve profiles at two chunk sizes. Boundary
tests reject inventory ceilings and one byte below combined residency, admitting the exact bound.
This guarantee applies to the transport-aware bridge, not numerical-only `execute`/`run_thermal`
or conversion of an already computed result. Portable native worker routing, scheduler/receipt
integration and whole-process qualification remain required.

The isolated `thermal_execution_residency` integration test measures the complete public bridge
above caller-owned model/mesh storage. Tet4 at 8/128 samples measured 203,014/2,973,342 B against
4,129,474/33,981,668 B admission bounds; Tet10 measured 203,691/2,979,779 B against
4,745,367/34,616,281 B. These small insulated fixtures demonstrate the tested requested-heap
envelope, not large-mesh, allocator/native-memory or portable-worker qualification.

The shared kernel handoff now distinguishes whole, partitioned and already prepared outputs with
a closed payload enum. `AnalysisKernelInvocation` carries the effective (default-capped) chunk size
and inventory limits. Prepared envelopes move through the serial bridge without gathering or
rechunking buffers; the projection includes the boxed typed envelope. Before publication the bridge
checks the canonical root bytes, canonical logical names, namespace/media, each chunk digest and
length, per-partition chunk boundaries and digests, whole-payload digest, inventory limits and the
declared resource envelope. Resource plans are decoded before kernel invocation. No output objects
are written until all output closures and the output-set envelope pass validation. Tests retain
buffer pointers across the handoff, reject substitutions and wrong policy, and compare published
whole/prepared outputs. The conversion-memory fixture includes this validation pass. This removes
the forced all-field buffering at the kernel boundary; thermal native dispatch now uses this path.
With prepared handoff validation included, the isolated 16/64-partition fixture measured
8,607,519/34,424,943 B of requested heap growth against reservations of
8,698,192/34,582,960 B. This remains requested-heap evidence, not a process-memory ceiling.

Thermal numerical receipts can also be constructed from a validated result header. The full-result
and header paths use the same borrowed metadata and produce identical canonical receipt bytes.
The bridge checks the header payload digest and its complete semantic partition inventory without
gathering field arrays. This receipt attests accepted numerical metadata, not the availability or
integrity of every field body; publication must still validate the complete transport closure.
It is not an independent accuracy assessment. Tests cover Tet4/Tet10 steady/transient parity,
workload substitution, altered header bytes and missing inventory entries, and header-only access.

Thermal host resource planning now has a dedicated workload-bound phase and demand (resource
estimation revision 26), rather than using structural operator facts. It consumes the numerical
preassembly reservation and adds decoded inputs, retained input transport, authorizing plan
storage, result and receipt construction, the shared output-set root, and worker overhead through
shared monolithic host accounting. `prepare_thermal_inputs` derives the input and output facts
through the bridge's canonical-object and transport projectors at the effective chunk policy.
The lower-level planner accepts supplied prepared facts; worker integration must rederive these
from immutable inputs rather than trust submitted byte counts. Only an explicit CPU backend is
admitted by this new plan builder. Tests cover source/phase binding, canonical plan round trips,
additive receipt storage, exact/one-byte-short memory ceilings and undersized output inventory.
These are planning tests, not a measured portable-worker admission claim.

The `SolveThermal` host task now carries the canonical workload, mesh and resource plan. Native
dispatch validates their identities and selected step, rederives the plan at the effective output
policy, and checks the execution identity before solving. It returns a moved prepared result and
a source-bound numerical receipt. Result import validates the exact returned reference inventory,
workload, topology and receipt without gathering field arrays. Terminal-temperature publication
uses the thermal manifest's semantic result identity rather than a host/transport-derived identity.
Resource-planned task submission uses the shared analysis allocation check. This path is an
explicit-CPU formulation envelope. The native process-conformance harness submits
Tet4/Tet10 insulated constant-temperature transient tasks through the shared resource admission
and isolated, durable, and loopback TLS/QUIC workers, then checks exact inline/worker result and
receipt parity and analytic equilibrium across nonempty field partitions. Each placement also
publishes and imports the terminal temperature state. Durable tests reopen the supervisor after
completion before importing outputs; they do not establish interrupted-solve restart. The remote
fixture binds its actual run authorization scope and frozen-project program revision. This
establishes these execution paths for those fixtures, not general thermal accuracy or a distributed
thermal solve. Measured worker residency, interrupted-solve restart, and independent
accuracy/release gates remain unproven.

Thermal numerical receipts feed the shared accuracy evaluator through a typed thermal source.
The bridge binds the receipt to the numerical workload input, mesh and producer step, not to a
transport-dependent host identity. The only receipt-backed criteria currently supplied are
global-equilibrium algebraic residual bounds and conservation balances, using the maximum accepted
linear and energy residuals respectively. Their normalization assumptions are retained in the
evidence. These criteria do not establish spatial, temporal, recovery or reference accuracy.
Unsupported mandatory criteria remain indeterminate and follow the shared accuracy publication
rule. Satisfying a residual-only plan does not establish spatial or temporal accuracy.

`import_thermal_completion` imports the validated result closure once and reconciles it with
scheduler-owned solve-attempt accounting. It checks exact encoded artifact bytes, the admitted
thermal plan, allocation and semantic result identity before preparing shared resource-usage
evidence. The caller must retain the scheduler's response/accounting association: an accounting
record alone contains no task identity and cannot prove task lineage. CPU and memory usage here
represent reserved resources over the measured interval, not sampled utilization or peak RSS.
The shared result-publication path binds this evidence to the thermal accuracy report. Accuracy
evaluation now has its own source-bound resource phase (estimation revision 27), shared across
thermal and structural receipt sources. A borrowed shape projection counts criteria, matching
evidence, pre-deduplication assumptions and canonical report construction before evaluation; it
does not compare tolerances or generate the report on the driver. The worker rederives the plan,
and shared output admission enforces the report envelope. Evaluator collections are preallocated
from checked counts and sorted in place. These are requested-heap admission estimates, not a
measured process-memory ceiling.

Measured solve, accuracy and semantic-state attempts can be retained in publication usage.
Extending usage preserves all previously measured phase plans, including observation work.
Final result-publication tasks still lack their own FEA phase plans, so this is not complete
study-graph qualification.
Publication's own final interval must ultimately live in workflow-completion accounting, not in
the prebuilt source-usage object consumed by that same publication attempt.

Semantic-state output sizing now has borrowed encoding projections for all six supported
publication sources. Thermal sizing reads the result manifest and terminal field shape, not
temperature partitions; structural sizing borrows the producing worker's existing typed result.
The projections use the shared canonical profiler and are checked against actual state encoding
for primary displacement, linearization, modal basis, response envelope, random statistics and
Tet4/Tet10 thermal states. Primary field selection is shared with actual publication. The shared
state-publication plan now reserves retained input transport, decoded source residency, a separate
source-validation workspace, owned state materialization, and encoded output transport.
These are conservative codec-based admission estimates, not measured whole-process memory.
Structural planners must not decode full results on the driver merely to obtain these profiles.

All six supported source families emit a separate `StatePublicationProfiles` artifact. Its bounded,
sorted catalog binds each exact authored publication request and encoding profile to the producer,
topology, semantic result and source payload digest. Thermal import checks the complete output
inventory, then independently reprojects the catalog from metadata only; hashes alone are not
treated as proof of a claimed size. Structural importers independently reproject from the typed
result they already import. Resource estimation revision 29 reserves each producer catalog from
the actual declaration inventory. The publication worker rederives its resource plan and compares
the catalog entry with the actual source profile before constructing a state. Native schedulers
admit those plans and retain actual publication attempts; durable closure reconstruction explicitly
records a durable import without a live scheduler attempt. FEA execution is admitted and accounted
at its owning run, separately from this import metadata. Existing numerical receipts
and numerical result identities are unchanged. This does not qualify interrupted solves, complete
workflow accounting, or a broader formulation envelope.

The release `thermal_history_memory` diagnostic separately measured identity-construction growth
above the retained-result baseline: 331,318 B for connected Tet4/257 samples and 5,231,479 B for
connected Tet10/4,097 samples (both unforced constant-temperature references, zero temperature
error). These are measured Rust-heap cases, not complete producer residency bounds.

Numerical-object chunk preparation hashes borrowed partitions incrementally rather than
concatenating them into an additional payload buffer. Closure verification checks existing chunks,
partition digests and the complete digest directly, without gathering and rechunking. Canonical
chunk boundaries and payload identity are unchanged. Invalid inventories are rejected before indexing.

Run the Rust-heap diagnostic with:

```sh
cargo run --release -p runmat-analysis-fea --example numerical_chunk_memory
```

For sixteen partitions and 64 KiB chunks, an 8 MiB payload required 8,400,432 B additional
Rust heap during preparation, and a 32 MiB payload required 33,593,904 B. Verification required
1,200 B for both cases (partition-identity metadata). Input construction is excluded; preparation
includes the owned output chunks. This is not a complete producer/transport memory bound:
thermal result serialization, bridge import/materialization and retained producer payloads require
separate measurement. Portable thermal Solve routing now exists, but routing conformance does not
prove these residency bounds. Reported process RSS is contextual only.

Bridge whole-object and selected-partition imports also hash their retained chunks directly,
without a verification-only aggregate buffer. Exact chunk lengths and validated manifest sums
prove aggregate lengths; full import verifies every partition and the total digest, while partial
import verifies only the selected partition and does not read unrelated chunks. Import still owns
the selected chunks. Explicit `payload()` calls and canonical resource-profile materialization
still gather bytes; this change does not establish a whole-pipeline memory bound. Segmented import
does not require the aggregate length to fit one contiguous `usize` allocation.
