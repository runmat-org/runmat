# Structural adaptation from original solved results

This is an in-process evidence and acceptance path. Stage 11 and the
original cantilever convergence criterion remain open.

## Original equations and fields

Structural static results now retain typed convergence: iteration count, absolute
residual norm and relative residual in the actual equation space. Ordinary PCG,
affine-reduced PCG and multiplier GMRES preserve the same reference norm used by
the original solve; completed accelerator evidence supplies its independently
recomputed relative residual. Invalid admission has no relative residual. A
balanced zero equation may converge in zero iterations. These values are distinct
from recovered physical force/work balance. Static cold imports require the typed
carrier; other procedures reject it. An absent static carrier is omitted from
nonstatic serialization, preserving existing immutable publication record digests. Runtime static acceptance consumes the typed
carrier rather than the severity or text of a convergence diagnostic. An unconverged solve
is numerically unacceptable even under exploratory quality policy.

`StructuralAdaptationSolvedSource` binds the original model, canonical mesh and
complete retained static result. The logical solver-result identity excludes backend
and attempt diagnostics; the separate original-content digest binds the complete
returned result. It resolves the authored final scalar observation
through the shared observation owner and checks it against original retained
fields. Displacement components and reductions keep their authored quantities,
frames and signs. It does not manufacture a scalar quantity by relabeling raw
field values. Shared canonical profiling admits encoding plus original-field
copy construction before payload copies; cancellation reaches the
shared codec. This component budget is not whole-task resource admission or RSS.

`build_structural_adaptation_iteration_from_solved_results` estimates original
integration-point stresses, transfers original displacement to the new mesh,
measures it against the actual re-solved target displacement, rebuilds constraints,
and feeds original solver and observation facts into shared study accuracy. Transfer borrows original components and retains only new projected values; it
does not copy every original nodal array into a second map. Shared transfer observes
cancellation before allocating and at bounded target-node checkpoints. The
previous solved target must be the next original source. There is no new solver,
scheduler or tolerance model.

## Evidence scope

Tests execute both original Tet4 and Tet10 public static routes, retain/cold import
results, and reject missing convergence, tampered summaries/fields and missing or
changed observations. A real Tet4 h-change adds interior DOFs and requires a fresh
solve. The independent affine reference has E=100 Pa, nu=.25 and u=(.02x,0,0),
with stress (2.4,.8,.8,0,0,0) Pa. Original source/target stress, displacement, transfer,
recovery, constraint lineage and shared accuracy are checked together. Component
materialization required-minus-one rejects; required bytes pass; cancellation
rejects. Hashes bind provenance; they do not independently prove equilibrium.

This does not replace the failed cantilever criterion or qualify a nonlinear
history transfer. Original-result revalidation requires original model, meshes,
fields and observations. Decoding a summary alone does not verify its provenance.
Durable/native adaptive tasks, producer termination, retries, restart, remote
execution, whole-graph resources/RSS and semantic publication still need evidence.
The current direct static route accepts its original single-step model. Family
estimators still require their independent scientific qualification.

## Transfer arithmetic and ownership

A red Tet10-centroid regression produced infinity although its exact interpolated
value is finite (`f64::MAX`). Projection now uses shared compensated arithmetic and rescales only after loss
of range. Constants and tiny surviving cancellation terms remain preserved; unsafe
loss of a nonzero contribution rejects. Nonfinite
original or unrepresentable projected values reject. Linear midside averaging uses a finite sum when available and scales operands
before addition only when that sum overflows, preserving subnormal means. Compensated accumulation moved unchanged from
observation internals to shared algebra; partition combination moved unchanged
into its own observation module. Existing module-size debt ceilings were lowered.


## Original native producer closures

The execution bridge now composes an adaptive decision from both original native
static solve closures: producing host contracts, exact invocation and mesh inputs,
complete original results, and their numerical receipts. It authenticates the
transport chunks, checks the original producer input inventory, validates each
receipt against the original result, and requires the same authored study, step,
physical model and artifact-access context. The prior target must be the next
source. The FEA owner evaluates recovery, transfer, constraints, observation and
shared accuracy; the bridge introduces no numerical operator or scheduler.

A shared codec scanner profiles canonical payloads across borrowed transport
chunks. Resource sizing no longer gathers every partition before checking its
decode admission. Contiguous and chunked profiling share one inventory algorithm,
validate split UTF-8 and scalar headers, reject unsupported/truncated data and
check cancellation within large strings as well as between chunks. These are
requested decoder estimates, not measured RSS or whole-task admission.

The native witness performs the original coarse and refined affine Tet4 solves
through the existing native kernel and filesystem object store. It checks the
independent stress reference, original result/receipt identities, shared accuracy,
cold summary revalidation, exact decoder-budget boundary, receipt substitution,
foreign artifact access and reversed chaining. Reopening the original objects
from disk verifies source imports after the producing store is closed; this is
not process-separated producer termination. A summary cannot be independently
verified without its original sources.

A separately scheduled adaptive worker task, complete evaluation resource plan,
closed adaptive output/root, atomic semantic publication, durable progression and
remote execution remain pending. The process witness below addresses producer
termination and cold replay separately. The affine witness
also does not resolve the original cantilever convergence gate, qualify nonlinear
history transfer or close Stage 11.


## Producer termination and cold consumers

A process witness now prepares the original native requests and writes their
immutable input objects without executing a solve in the parent. A separate child
executes both requests through the existing native analysis kernel and returns the
original externalized completion references. The parent observes and reaps that
exact producer before launching either consumer. Fresh consumers import the
request-bound original result/receipt closures from the filesystem and evaluate
the shared adaptive evidence without invoking a solve. Independent checks cover
all original nodal displacements and six integration-point stress components of
the affine patch. Two consumers retain identical complete iteration evidence.

Required decoder bytes pass; required-minus-one, cancellation, reversed source
order, a missing original solver-result chunk and a corrupted chunk reject without
writing an evidence file. Transport faults use a test read adapter and leave the
original immutable artifacts intact. The test protocol has bounded canonical reads
and a finite child lifetime; subprocess stdout/stderr goes to files with bounded
inspection. Closing a store alone is no longer the sole cold-execution witness.
This is a Mac native process witness, not a scheduled adaptive graph, remote solve,
measured RSS or atomic semantic result publication.

Source materialization admission now runs before constraint lowering and observation
plan construction. The regression first reproduced the old late rejection and
passes with the unchanged byte formula and ceiling. The component allowance covers
canonical encoding and retained-field copies; it excludes the separate construction
workspaces of constraints and observation plans. Native consumers use the shared
borrowed input profiler directly and do not copy eight original payload closures
just to size their decoder inventory.

Complete pre-expansion constraint-lowering profiles and the adaptive evaluation
working inventory are prerequisites for complete scheduled-task admission. The
shared constraint compiler currently accumulates fallible rows but explicitly lacks
that pre-expansion profile. That gap remains open, along with the scheduled task,
closed adaptive output/root, durable progression, semantic publication and all
original scientific qualification criteria.


## Counted shared pairing workspace

Tie and periodic pairing now derive a checked requested-slot profile before building
coordinate indexes or pairing evidence. The solid and mixed paths use compact sorted
vector indexes with fallible exact reservations, replacing copied tree maps and the
matching-used tree. Facets stream from the original mesh; transformed facet nodes and
interpolation values use fixed stack arrays. Geometric weights are allocated only
when a candidate replaces the retained best facet. The profile includes retained
weights and both overlapping candidate buffers, including all six quadratic weights.
Pairing row and term buffers also have checked counts and fallible reservations before
expansion. Existing geometric formulas, thresholds and canonical identities remain.

`constraint_pairing_workspace_profile` exposes the solid pairing inventory; the mixed
owner derives the same inventory with its actual canonical node count. This is requested
vector storage, excluding borrowed sources/targets, allocator rounding/bookkeeping,
strings, row construction/encoding, fixed transform stack scratch and measured RSS.
It is not a whole-constraint or whole-adaptive-task admission receipt, and no task cap
or resource-policy authority is inferred from successful allocation.

Cancellation reaches counting, index construction, matching candidates, facets and
row expansion. Duplicate node-index keys and inconsistent counts reject; empty targets
reject instead of indexing an absent secondary. Shared constraint canonical-byte goldens
remain unchanged. An independent nonconformal Tet10 surface trace checks quadratic
weights, negative coefficients, every displacement row and identity preservation under
node/facet reordering. Original acoustic periodic field-convergence references and
native cold-process checks still pass. No original scientific gate is replaced.

At the pairing checkpoint, remaining pre-expansion work included target selection,
normals, coupling and reference points. The shared construction slice below addresses
those components. Complete row encoding, topology, graph/validation and task storage
composition remains required before whole-task reservation can be claimed.


## Counted target, normal and coupling construction

Shared constraint construction now exposes checked requested-storage profiles for
nodal target selection and metadata copies, original normal recovery, solid coupling
and reference-point preparation. Borrowed entity candidates are deduplicated before
cloning unique metadata. Counted sorted vectors replace raw-node and coordinate tree
indexes and coupling measure maps. Fallible exact reservations precede expansion;
overflow and cancellation reject through the shared typed constraint error.

Normals reuse one buffer for accumulation and normalized output. Projection storage
uses the existing topology owner's construction estimate; facet coordinate and vector
scratch are counted separately. Original area-vector recovery, curved-facet orientation,
subtriangle coupling weights, geometric thresholds and rigid/distributing equations
remain unchanged. Explicit-surface coupling borrows the shared shell-node coordinate authority;
its counted construction inventory is described below.
Coupling equations and rotational checks are in focused modules under constraints.
The superseded pairing-only node index is removed in favor of the shared compact index.

Reference points clone only unique trimmed IDs. Duplicate IDs retain the original
last-authored position, including the signed-zero case where numeric equality alone
would hide a canonical identity change. Existing mixed solid/beam/shell canonical-byte
goldens pass. Independent tests cover Tet4/Tet10 right-triangle area weights, outward
normals and every rigid/distributing equation against a prescribed rigid motion.
Cold native producer-termination and original-result replay remain separate evidence.

These are component requested-storage profiles. They exclude borrowed inputs,
allocator rounding/bookkeeping and RSS. Row/coupling/target identity encoding and row
stable-sort scratch are separate; the normal projection quantity is the shared owner's
estimate, not exact allocator instrumentation. Complete compilation also needs mixed
and explicit-surface topology construction, row/plan encoding, accumulation/sorting,
graph/rank/validation aggregation and adaptive evaluation/output composition. No
whole-constraint/task admission, scheduled adaptive graph, atomic semantic publication,
measured resource envelope or original scientific gate is qualified by these profiles.


## Counted ordering and canonical finalization

Constraint sorting now uses a counted ordinal vector with fallible reservation and
in-place permutation cycles. Original ordinal is the tie break, preserving stable
term order and floating-point duplicate-coefficient summation. Source-equation hashing
borrows targets rather than cloning their strings and persistent-entity metadata.
Authored source preparation profiles its source/ID/digest/ordered-reference vectors and
shared canonical materialization estimate before constructing those buffers. Plan
accumulation requests exact additional capacity instead of geometric growth slack.

Row finalization exposes its additional ordinal, possible compaction, ID-copy and
canonical materialization inventory. It excludes the already-owned incoming term buffer;
that allocation remains the lowering owner's responsibility. The row codec shape uses
all original candidate terms and the longest recovery/residual enum spellings so that
post-combination singleton or rotational rows remain covered. Estimates are supplied by
the shared canonical codec, not a private string/map multiplier or measured RSS.

Canonical validation now borrows sorted reference points for binary lookup and uses
sorted row ranges for group counts, removing copied reference sets and row-reference
vectors. A focused validation module owns the existing semantic checks. Checked evidence
addition rejects overflow; every row must belong to a declared group. Hostile canonical
payloads bypassing producer validation are rejected at cold import for both cases.
Valid canonical plans, original rank/conflict tolerances and numerical equations remain.

The post-lowering plan finalization profile composes counted group/row ordering, shared
graph preflight and component-bounded rank workspace, plus the shared full-plan codec
estimate. Its graph derivation allocates counted graph scratch: the existing count-only
graph reservation must be admitted before requesting the tighter profile. This is a
phase inventory, not a pre-lowering whole-compilation receipt. Mixed/explicit topology,
complete lowering/evidence construction and overlap, adaptive evaluation/output, scheduled
execution, semantic publication, measured resources and scientific qualification remain
separate requirements. No resource ceiling or original failed acceptance target changed.


## Counted explicit and mixed topology construction

One analysis topology owner now supplies fallible counted sorted indexes to topology,
constraints and their surface consumers. Explicit identity validation uses authored
ordinals to preserve duplicate/nonfinite/missing-node error precedence. Its requested
validation inventory counts actual reference/index vector slots rather than tree-node
multipliers; resource estimation revision is81. Canonical topology and element bytes
remain unchanged, including signed zero and attachment metadata.

Solid, explicit and conformal mixed node-space construction profiles requested nodes,
explicit indexes, attachment ownership and uniqueness scratch before allocation. The
inventory composes the existing mesh validation and shared codec estimates, including
mixed identity materialization. Attached nodes upgrade original solid ownership in
place; unmatched explicit nodes retain authored order. Attachment resolution streams
its match count, allocates labels only for errors and rejects duplicate ownership.
The focused constructor and resource modules replace the oversized node-space parent.

Shell surface construction retains one sorted node record for original coordinates,
nodal area measure and normal, plus authored-order triangles with borrowed labels.
The inventory includes overlapping raw candidates and retained records. Loads,
connectors, normal supports and reference-point coupling consume this shared geometry.
Area thresholds, cross products, authored accumulation order, first normal signed zero
and normalization thresholds remain original. Mixed target selection uses counted
vectors with sorted/deduplicated canonical indexes; initial-state and constraint callers
move their owned node selection rather than clone it.

Constraint codec profiling now separates a conservative size bound from the actual
wire limits. A streaming small-integer inventory proves the upper bound can exceed
64MiB while the actual encoding remains below that unchanged limit. Actual encoding
and cold decoding still enforce their original limits; no payload limit was raised.

Independent references cover orthogonal shell areas/normals/weights, mixed ownership,
exact canonical identities, error precedence and cancellation. The original shell and
resource regressions remain required. These component requested-storage inventories
exclude borrowed inputs, allocator overhead and RSS. Complete lowering/evidence overlap,
pre-expansion whole-compilation admission, adaptive working/output composition, scheduled
execution and atomic semantic publication remain separate gates. The original cantilever
refinement failure remains open; no numerical tolerance or resource cap changed.


## Counted constraint row expansion

Solid, explicit and mixed lowering now return one shared compiled-constraint carrier;
the solid-only tuple conversion is removed. Ordinary fixed, prescribed, motion, normal,
symmetry and antisymmetry rows derive checked row/term/identifier inventories before
allocating their row buffers. Motion counts use checked multiplication instead of
saturation. Equation target storage uses fallible exact reservations. Prepared target
occurrences now supply the complete term count before equation expansion. Existing pairing/coupling owners retain their separate counted inventories.

Singleton and directional term vectors reserve their actual nonzero term counts.
Row/group identifier copies and group reference-point identity vectors are fallible;
short target inventories move into exact buffers. Normal/tangent directions use fixed
arrays, and mixed DOF traversal uses an iterator instead of per-node allocated vectors.
The original basis transforms, equations, rotational ownership, floating-point term
combination and canonical-byte goldens remain unchanged.

Mixed family routing now propagates target-resolution failures instead of treating them
as a request to fall through to mixed lowering. Cancellation remains typed through
node-space dispatch. Invalid inputs may consequently report an unresolved target earlier.
Independent references retain fixed Tet10 translations and explicit shell rotations,
original canonical replay, requested retained-row inventory bounds, overflow and entry
cancellation. Analytic normal/reaction, curved-foundation, rotated-periodicity, distributing
wrench and accepted nonlinear kinematics tests remain mandatory.

The retained row expansion profile counts row/term slots and identifier payloads. The
composed construction profile now additionally includes the largest overlapping row
identity/sorting/compaction workspace. Both exclude target geometry, plan accumulation/
classification, allocator overhead and RSS. Equation profiles derive retained term counts
from prepared selections before allocating the equation term buffer. Target preparation itself is still additional
construction work outside this row-only inventory. Group/evidence construction is also
additional to this row inventory.
It is a component inventory, not whole-compilation policy admission. Full pre-expansion
composition and retained-output overlap remain required for the scheduled adaptive path;
no original scientific gate, resource cap or qualified envelope is promoted here.


## Prepared constraint target occurrences

The shared compiler now prepares each authored target occurrence before family dispatch.
Its owned selection carries resolved provenance and canonical node indexes into lowering;
solid, explicit, shared and mixed routing consume that same ownership evidence. The old
resolve-and-discard routing pass is deleted. The existing mixed-topology ownership and
constraint-selection resolvers retain their separate semantics; this change removes
repeated calls across routing/lowering, not their distinct checks during preparation.

Lowering moves the original selection buffers, retains authored equation occurrence order,
and rejects exhausted, reordered or unused occurrences. Prepared selections remain scoped
to one constraint; there is no global duplicate target cache. Equation terms count all
selected node occurrences before expansion, including duplicates which the existing row
owner subsequently combines in its original order. Empty explicit equations reject at
the original typed lowering gate without dispatching to an absent solid mesh.

Original plan byte goldens and analytic constraint tests remain mandatory. Invalid inputs
can report a later target failure during preparation before an earlier coefficient or
value failure; supported valid equations and transfer provenance are unchanged. Preparing
all occurrences introduces target-buffer overlap which must be composed into the full
compilation demand. Fallible counted reservations and pre-term row sizing are not a whole
compilation budget receipt. Complete admission, adaptive output overlap, scheduled execution,
atomic semantic publication, measured resources and the original scientific failures remain
open; resource revision stays81 and no tolerance or cap changes.


## Composed row-construction workspace

All ordinary solid/explicit/mixed row buffers and equation term expansion now derive
retained row/term/identifier storage together with the largest overlapping row-finalization
workspace before allocating the row buffer. The row owner supplies ordering, possible
replacement storage during compaction, identifier copying and shared canonical codec
materialization. One count-only serialized identity shape bounds existing node/reference
owners and translation/rotation terms without retaining a representative term vector.
Its inventory values never become numerical terms, row identities or physics evidence.
Actual row construction continues to use the original terms and identity encoder.

Rigid and distributing couplings use the same composed owner. Distributing coupling counts
the three original translation and three moment buffers separately rather than charging
six maximum-width rows. Tie/periodic rows count actual transform coefficient directions
and primary weight inventory before expansion; no blanket dense matrix assumption is used.
Empty row inventories reserve no finalization work. Overflow, inconsistent inventories
and entry cancellation reject before buffers; no floating-point equation or sum changes.

Tests compare the count-only finalization envelope with original serialized term shapes
for both owner classes and DOF families, tiny and maximum finite coefficients, Unicode
identifiers and up to1024 terms. Existing actual canonical materialization, exact byte
goldens and analytic constraint references remain mandatory. This composition is requested
row-construction storage and addressability, not whole-compilation policy admission or RSS.
Prepared-target overlap, geometry, group/evidence retention, plan accumulation and component
rank still need complete phase admission before adaptive output and scheduled publication.
Resource revision stays81; original scientific failures and acceptance limits remain open.


## Group evidence and retained accumulation workspace

Group construction now derives additional target ordering, ID-copy, optional reference-ID
and authored source-digest workspace before its buffers. Whole authored preparation and
individual group source digests share one source profile, including borrowed equation-term
references, stable ordering and original source materialization. Their numerical identities
and canonical source encoding remain unchanged.

The compiler profiles original completed group evidence and incoming rows incrementally
before row-buffer reserve/append. A single collection envelope is retained; per-increment
structural bytes come from the shared codec's envelope-aware counter. Existing accumulated
rows/groups are neither cloned nor rescanned for each increment. The inventory includes
planned group records and possible overlap of old/new/incoming row-record allocations;
term payload ownership moves without cloning. Arithmetic, capacities and addressability
are checked before reserve. Group construction precedes append, preserving successful
plan identities and original physical fields.

Tests retain borrowed equation slots, source/group composition, incremental bounds against
a combined original Tet10 plan, immutable canonical bytes, overflow, inconsistent capacities
and cancellation. These are requested storage inventories and shared materialization
estimates, not RSS. Group construction excludes already owned lowered evidence; accumulation
starts after lowering. Prepared-target/geometry work and finalization/rank phase overlap
still need full policy admission before the adaptive task/root/publication envelope can be
qualified. Resource revision remains81 and original scientific acceptance remains open.


## Prepared target selection construction inventory

Explicit beam/shell region selection uses the shared sorted authored-node index instead
of rescanning every authored node for each endpoint. Point selection, original endpoint
membership, canonical sorting and deduplication retain their valid-input semantics.
Duplicate authored node IDs now fail at index construction, including an absent selector;
canonical structural topology already rejects this invalid input. Missing endpoints retain
the helper's original behavior and remain subject to owning topology validation.

Count-only explicit and mixed profiles include requested index slots and raw selection
candidates before index/selection allocation. Mixed solid candidate counting shares the
actual selector's provenance, boundary-face and volume-element membership rules; it does
not materialize a temporary selection merely to count it. Prepared occurrences compose
these buffers, explicit node identities, authored target metadata, eligible solid-selector
workspace and occurrence records before reserving the per-constraint occurrence buffer.
Checked arithmetic, addressability and cancellation precede those reservations. Authored
occurrence order and original solid/explicit/shared ownership remain authoritative.

Independent beam/shell endpoint-selection references, both-order repeated occurrence
checks, original mixed target ownership, duplicate-ID rejection and cancellation cover
the new construction path. Original canonical plan byte goldens and analytic constraint,
resource, periodic and native cold-field tests remain mandatory. Profiles conservatively
include alternative solid/explicit selection buffers; they do not assert measured RSS.
The shared identity inventory below fills the prior selection-encoding exclusion.
Topology/node-space, lowered geometry, rows/groups, component rank and finalization
remain additional phases. This inventory is not complete
compilation policy admission. Full adaptive working/output admission, scheduled execution,
atomic semantic publication and original scientific qualification remain open. Resource
revision stays81; no acceptance limit, mesh or resource ceiling changes.


## Shared selection identity and codec cancellation

Solid and explicit selection, and cold selection revalidation, now use one analysis-owned
selection identity contract: the original authored target, original matched entities and
original node identities under the original domain. Actual tuples borrow their buffers;
explicit empty provenance is a borrowed empty slice. Selection digests and plan wire bytes
remain unchanged. The count-only selection inventory includes the shared codec's canonical
materialization estimate before selection allocations, in addition to the existing index,
candidate, identity-copy, metadata and occurrence slots.

Candidate provenance is streamed from original mesh/target objects without a temporary
entity vector or copied strings. Raw duplicate entity occurrences conservatively bound the
deduplicated identity. A streamed maximal digest-width inventory bounds node identity
encoding; those inventory values never enter physical fields, selection digests or persisted
state. The actual borrowed tuple is also profiled before encoding. The codec's shared
estimate and addressability checks are used; no private wire-size multiplier or allocator
claim is introduced.

The owning constraint codec now carries the existing FEA cancellation token into shared
canonical encoding and cold decoding. Source, row, reference-point, selection and plan
identities use that same controlled encoding entry. Typed cancellation is retained at the
domain boundary; other codec errors retain their encoding classification. Profiling checks
cancellation before and after its borrowed traversal. The shared codec checks before and
after Serde projection and during canonical value processing; arbitrary custom serializers
are not preempted inside their own execution.

Tests cover original contract bytes and cold values, unchanged digests, both element orders,
face provenance deduplication, exact/numeric points, Unicode/long metadata, zero through1024
node identities, overflow, entry cancellation and cancellation raised during profiling or
Serde projection. Original compiler byte goldens and public/native field witnesses remain
mandatory. This completes the requested selection-identity component inventory, not whole
compilation admission, measured RSS, scheduled adaptive execution or atomic publication.
No scientific tolerance, mesh, resource ceiling, schema or estimator revision changes.


## Composed compiler preparation inventory

The compiler derives a preparation inventory before allocating its authored reference list,
topology/node-space, reference-point or prepared-target buffers. It composes the shared
authored-source profile, the original topology owner's validation/construction estimate,
reference-point records/IDs and identity materialization, and the largest per-constraint
prepared-target inventory. Only one constraint's prepared occurrences are retained at a
time; rows/groups accumulated from earlier constraints are a separate later-phase demand.
Preparation sums transients and retained buffers conservatively even when lifetimes differ.

A borrowed typed input discriminator selects the original solid, explicit or mixed route
once for both resource profiling and construction. It does not copy node-space state or
replace topology validation. Reference-point preparation now includes the largest original
trimmed-ID/position tuple's shared canonical materialization estimate before candidate and
point buffers. Raw repeated IDs bound record/copy storage; original deduplication, conflict
checking, authored overwrite order and signed-zero behavior are unchanged.

Both-order tests compile and cold-round-trip solid, explicit and mixed plans, compare actual
construction topology identities and leave original models unchanged. Repeated reference
IDs with long Unicode payloads check actual retained storage and original identity workspace.
Empty and cancelled preparations reject or retain zero target workspace as appropriate.
Original canonical byte goldens and numerical/cold native witnesses remain mandatory.

This remains a preparation inventory composed with existing shared estimates, not a whole
compilation policy receipt or measured RSS. The initial preparation slice exposed the shared meshing validator's approximate tree-storage
allowance. The subsequent counted-validation slice below replaces that path in its owner.
Other projection and compilation phases still require their own complete accounting. Lowered
geometry, rows, group retention, component rank, finalization, adaptive output and scheduled
publication remain additional governed phases. Invalid inputs can report a target, topology
or reference encoding failure earlier than the former ordering; supported plans retain the
original equations, identities, fields, meshes, tolerances and caps. No stage promotion.


## Counted shared mesh validation

Shared mesh validation now uses canonical numeric order for borrowed node, element, face
and region lookup. Fixed-size connectivity/provenance checks do not allocate temporary
sets. Stable identity uniqueness and reciprocal neighbor keys use explicitly reserved flat
buffers with in-place sorting. Tet10 edge occurrences reserve exactly six entries per
volume element, check shared-midpoint consistency and retain capacity after deduplication.
Node role and region/contact/interface classification use counted byte flags.

The mesh owner checks requested slots and multiplication before construction, checks
cancellation during construction and around sorts, and reports allocation failures as
meshing contract errors. Sorting is not preempted internally; this is not a measured
maximum-latency guarantee. Existing validity requirements, fields, identities and canonical
serialization remain unchanged. Invalid inputs may fail earlier during resource or batch
identity checks. No private FEA mesh validator or generic tree approximation remains in
this shared artifact-validation path.

The owning scratch inventory combines retained ordered ID vectors with the maximum live
identity, order, neighbor or classification phase. Canonical identity materialization is
a later phase profiled by the existing shared codec. Requested layout bytes exclude the
borrowed input, stack, allocator overhead, diagnostic error strings and RSS. A test-only
thread-local allocator independently reconciles execution: all eight canonical, connected,
three-region and48-element slab fixtures across Tet4/Tet10 match their requested phase
peak exactly and release all scratch. Requested peaks range312–8930 bytes in these fixtures;
these are validation scratch, not whole mesh/solve memory or scalability qualification.

All120 meshing-core tests pass, including malformed-reference/identity/classification cases,
sparse numeric IDs, late cancellation and unaddressable slots. The full tetrahedron292 and
surface55 test scopes pass. FEA constraints142, resource171, periodic3, bridge200 and native
cold2 pass on the combined candidate. Original adaptation remains35PASS/1FAIL at the same
unchanged cantilever refinement criterion. Estimation model revision82 explicitly versions
the changed shared contribution; schemas, authored caps, meshes and scientific targets are
unchanged. Whole compiler lowering/retention/rank overlap, adaptive output and scheduled publication
admission still require closure. Analysis projection storage is addressed below. No stage
qualification or measured whole-task memory claim follows from this component evidence.


## Counted analysis projection and native field validation

AnalysisProjection now reserves sorted node/element indexes and raw directed neighbor
occurrences using checked counts. Neighbor keys sort/deduplicate in place without changing
capacity; compact per-node adjacency is separately reserved. Bounded element/face coordinates
use stack arrays. Original topology order, adjacency, identities and numerical integration
remain unchanged. Sealed mesh validation uses the shared meshing cancellation-aware path.

Constructor accounting includes actual typed records, connectivity, raw neighbor occurrence
overlap and shared canonical materialization phases. Clone accounting covers typed buffers
and original metadata payloads; it no longer charges serialization envelopes for a clone.
Independent test-only allocator checks bound constructors on six original Tet4/Tet10 fixtures
and match clone requested bytes exactly. The48-element Tet10 fixture has1,347,292 requested
constructor peak bytes below2,543,398 estimated, and66,976 clone bytes exactly. This is
requested Layout storage, excluding allocator overhead/RSS and whole-task admission.

FiniteVolumeFieldTopology validates uniqueness/references with three exact-reserved sorted
digest buffers. Its owner profiles those buffers plus one shared face-identity codec
materialization. Original stored order, owner-oriented face identity and wire payloads
remain unchanged; malformed duplicate/zero/reference cases reject. The tested native layout
requests12,576 peak bytes versus86,018 estimated and releases all scratch. Count-only codec
width inventory never becomes an original physical identity. No new field-validator
cancellation or maximum-latency guarantee is claimed.

Final combined candidate: core121, topology72, constraints142, resources171, fields28, CFD43,
periodic3, bridge200, native adaptation2 and native fluid8 tests pass, plus three producer child executions. Strict scoped lint
passes. Adaptation retains35 passes and the unchanged original cantilever refinement failure;
all12 recorded scientific failures remain open. Revision83 versions resource contributions,
without changing authored caps/meshes/targets or schemas. Full compiler/adaptive/scheduled
admission, scientific qualification and attested release remain mandatory.


## Prepared lowering inventory

The shared compiler now composes additional row construction and geometry workspace before
consuming its original prepared target occurrences. It borrows selections without resolving
them again. Solid, explicit and mixed routes use the same ownership decision as lowering.
Fixed mixed rows count actual rotational ownership; repeated equation occurrences are kept
uncombined until the original row owner processes them. Prescribed/normal/tangent rows use
counted shapes with the existing normal and explicit-shell geometry inventories.

Rigid/distributing rows reuse the coupling owner's shape; its coordinate/weight/lever-arm
inventory is separated from row storage. Pairing uses the original selected facet width
and the existing displacement transform's exact nonzero decisions. No pairing, geometry
replacement or equation construction is performed by this profile. Some phase inventories
are summed conservatively; this is requested-storage/addressability, not measured peak/RSS.
Prepared selections/topology retention, group accumulation and final rank remain separate.

Four new tests reconcile original row counts and term capacities across ten canonical
solid/explicit/mixed fixtures, both-order coupling/surface directions, repeated equation
targets and periodic transforms; cancellation leaves selections unconsumed. The original
canonical byte goldens and numerical constraint regressions remain unchanged. Final296
selected constraint140, resource171, bridge200 and native adaptive cold2 tests pass; strict
FEA lib/test lint, scoped formatting and diff checks pass. Adaptation retains35 passes
and the same original cantilever failure. The isolated architecture check retains the same
two excluded renderer debts. Full caller-policy compilation admission, adaptive scheduling
and atomic semantic publication are still required; no stage promotion.


## Caller-governed compiler component

`compile_for_model_with_workspace_limit` checks a caller-supplied additional-heap limit before
preparation, selection, lowering, group construction, row relocation, rank-profile graph
construction and finalization. The ordinary compiler delegates to the same implementation.
Planning/execution owns the policy; constraints owns phase lifetimes and quantities. Successful
construction returns fixed-size phase maxima without changing canonical physical plan bytes.
These records are not serialized proof or a cold-import authority.

Retained evidence includes counted vector capacities after target/row compaction and shared
canonical payload estimates. Incoming groups are traversed incrementally without rescanning or
copying earlier groups. Shared graph profiling is admitted before it allocates. Preliminary
preclassification counts only bound workspace; classification replaces them before validation
and successful return. Borrowed model/mesh, other caller-retained state, allocator overhead,
diagnostic strings and measured RSS remain outside this compiler component.

Direct static and nonlinear preassembly plus structural bridge preparation pass the existing
authored memory ceiling. A direct static rejection test exercises this through the public
preassembly path before duplicate-source lowering. Ten original solid/explicit/mixed fixtures
preserve original plans at the exact reported ceiling and reject required-minus-one. Independent
execution allocation peaks5962–176097 requested bytes fit inventories380970–997708 bytes. This
is fixture evidence for conservative requested storage, not allocator/RSS/scalability evidence.

Final297 constraints143, staticpreflight6, nonlinearpreflight4, resources171, bridge200 and
nativecold2 tests pass; strict FEA lint, formatting and diff checks pass. Adaptation retains
35 passes and the unchanged cantilever failure. The isolated architecture check retains two
excluded renderer debts; no owned debt/waiver. All12 recorded science failures remain open.

Whole task admission must still compose this component with caller-retained inputs and actual
allocated requests, route remaining recovery/frequency compiler callers, and govern later
encoding/observation/property/adaptive-output work. Scheduled adaptive execution, atomic
semantic publication, measured resources and attested release remain required. No promotion.


## Native retained-input composition and recovery routing

The static worker decodes and checks its resource-plan identity before recompiling original
constraints. Static and frequency solve/recovery workers derive original input facts, including
the resource-plan closure, before selecting the additional compiler limit. The limit is the
smaller planned memory request/authored ceiling minus original transport/decoded reservations
and fixed worker retention. Static validation releases its fresh compiler result before making
the solver sequence model copy. All canonical resource-plan rederivation checks remain active.

Static field recovery has no resource-plan input in its current contract: its compiler is bounded
by the authored ceiling minus original input facts. This is not a scheduled reservation claim.
Recovery-derived topology and later assembly/encoding/observation allocations remain separate.
The direct frequency recovery API also bounds its explicit compiler call by the authored ceiling;
its assembly path still requires complete phase composition. Frequency recovery is now a child
module with original function bodies preserved except for the compiler call; the former root
module-size debt entry is removed.

A real prepared native Tet4/Tet10 closure test includes the ResourcePlan input, composes retained
input storage with actual compiler construction, preserves the original plan at the exact combined
limit, and rejects required-minus-one and arithmetic overflow. Final298 strict FEA lint PASS;
resources171PASS, bridge200PASS, native adaptive3PASS, thermal/static1PASS and frequency8PASS/
1producer-only ignored. Cold frequency witnesses cover producer termination, original fields,
scalars/series, invisible precommit loss and retry. Selections overlap; no aggregate count claim.

The broader FEA frequency selection is47PASS/1FAIL: the original curved Tet10 one-DOF modal
reference stops at local-energy admission (9936 requested versus3520 reserved bytes), before its
physics verdict. The proof envelope is derived from the reduced matrix while retaining original
source/projection storage; this resource-policy boundary remains unresolved. It is additional
blocking evidence, not a newly qualified or counted scientific result. The12 recorded scientific
failures remain open. Strict formatting/diff PASS; main architecture PASS, isolated architecture
retains the same two excluded renderer debts. No owned waiver or stage promotion.

This worker component does not add compiler demand to canonical task resource requests (revision83
unchanged), establish allocated-resource reconciliation, measure RSS, or close whole-task admission.
Those requirements and scheduled adaptive execution/atomic publication remain mandatory.


## Projected original-energy proof admission

The299 implementation resolves the298 curved Tet10 modal admission mismatch. A proof of a reduced
operator from an original ten-node local energy processes the original grounding graph, local
LDL/outward scratch, bounded32-entry proof cache and projected coefficient inventory. The generic
sparse factor envelope derived from the one-DOF reduced CSR is insufficient for those phases.
The source-energy owner now derives a distinct count-only component demand from original widths,
projection storage and reduced support. Planning reserves the same additional storage; the
already-reserved sparse workspace is not counted a second time. Revision84 versions this changed
resource contribution. Original capture storage and cache capacity share definitions with execution.

Projected source execution uses that demand; generic sparse factor caps and explicit budget tests
retain their prior policy. Original identity-projection behavior, element coefficients, outward
verification, physical references, tolerances, meshes and authored resource ceilings remain unchanged.
Count inventory is not a mathematical certificate. Cold artifacts cannot manufacture the proof.
The inventory includes conservative allowances for coefficient pairs absent from numeric CSR and
for caches; this is requested-storage/work accounting, not allocator overhead, measured RSS or
whole-task allocation reconciliation. General rigid-mode/unrestricted-solid admission is not promoted.

An independent ten-vertex complete-graph energy reduced to one DOF proves positive coercivity below
its exact9 eigenvalue and reports work/storage within the owner inventory. Invalid width, overflow
and cancellation reject. Existing corrupt/unanchored/exhausted-budget tests remain unchanged.
The original physical curved Tet10 frequency and independent harmonic phase/power references pass.
Final FEA source18PASS, curved2PASS, broadfrequency48PASS, resources171PASS, strict lintPASS;
original damped-modal matrix5PASS251.29s. Final bridge200PASS2.71s, native modal1PASS1.76s
(both element orders; cold basis/fields/receipts and missing-budget rejection), nativefrequency
8PASS/1producer-onlyignored41.17s including cold publication and process-loss/retry. All jobs
terminal. Main architecture PASS; isolated retains the same two excluded renderer debts.
Scoped formatting/diff and87owned/8255excludedHEAD/98preserved/indexempty audit PASS.
The12 original scientific failures, broader family/execution and release gates remain open.


## Immutable compiler inventory and native modal routing

The300 carrier keeps the original canonical plan and its construction workspace private and paired.
Callers can borrow either through read-only accessors or consume the carrier to obtain its original
plan. They cannot replace the physical plan while retaining a foreign workspace report. The carrier
is not serializable, a numerical certificate, or a cold-import proof. The ordinary plan-only compiler
still delegates to the same implementation and consumes the result; canonical bytes are unchanged.

Native modal execution now checks original input/step identities, derives all original closure facts
including ResourcePlan, and selects its additional compiler budget from the smaller planned request
and authored ceiling minus transport/decoded/fixed-worker retention. It recompiles the unchanged
model under that limit and compares the exact original plan before resolving observations. The
fresh validation carrier is released before later work. Canonical resource-plan rederivation stays
mandatory. This closes that worker's prior unlimited validation entry point; it does not add the
compiler phase to canonical task requests or establish actual allocated-resource reconciliation.

The original modal witness now tests both Tet4/Tet10 real prepared closures at the exact combined
input/compiler boundary and required-minus-one. Cold basis/field/receipt and missing source-budget
rejection remain mandatory. Final constraints143PASS, API compile-fail1PASS, resources171PASS,
bridge200PASS, native modal1PASS (both orders) and native adaptive/cold3PASS. Strict FEA and native lib/test no-deps lint pass. Scoped format/diff pass. Main architecture passes; the isolated candidate
retains the same two excluded renderer debts. All12 scientific failures remain open; no stage,
whole-task, measured-resource or release promotion follows from this component.


## Canonical original compiler phase

Static, nonlinear solve and nonlinear recovery resource demands now require
`constraint_compilation_peak_bytes`. Revision85 versions this contribution and retention of
original encoded input closures alongside decoded values. The opaque construction carrier
borrows immutable original model/mesh inputs; its private plan and workspace cannot be detached.
The companion static planner reuses that exact fresh construction. Equivalent inputs compiled
independently produce the same physical plans and resource requests; source addresses never
enter canonical identities. A foreign borrowed model or mesh cannot reuse another carrier.
The ordinary plan-only planning API recompiles under the authored ceiling and compares the
original plan instead of accepting a detached or decoded workspace as construction evidence.

Monolithic requests reserve the larger of the original compiler phase and later numerical
phases, plus live encoded/decoded input retention and worker storage. Compilation is not
charged as permanently live solver scratch. Operator-domain recommendations retain full-source
compiler and input storage per worker; they cannot divide it by the operator partition count.
Their phase composition is conservative, not cooperative/distributed execution qualification.

The bridge and native static worker carry original compilation through canonical resource
rederivation. Direct static/nonlinear preflight does the same; nonlinear solve and readonly
recovery conversions preserve the contribution. The native worker rejects a positive understated
compiler reservation even after its resource-plan bytes and identity are recomputed. Cold
imports require the field, but its decoded count remains a claim, not a fresh construction record.

Focused tests cover compiler-dominated peak memory, exact authored boundary/minus one, missing
or zero reservation, source-scope substitution, independently reconstructed identity equivalence,
nonlinear recovery conversion and both-order cold native rejection. Solver-specific transform
and saddle-GMRES sizing are extracted into their existing resource owner's child module; all four
production function bodies are unchanged except visibility. No operator, mesh, tolerance,
authored ceiling, constitutive state or field approximation changes.

This closes the canonical contribution for these three demand families. The modal and frequency
extension is verified below. Transient and other compiler callers still need their respective
canonical phase composition.
Actual allocated execution resources, later encoding/observation/property/adaptive working and
output phases, measured RSS and whole-task qualification remain mandatory. No stage,
distributed/GPU or release promotion follows from requested-storage accounting.

Final verification: constraints143PASS, staticpreflight6PASS, nonlinearpreflight4PASS,
carrier compile-fail1PASS, resources173PASS and strict FEA lintPASS. Frequency48PASS and original
damped-modal5PASS339.76s retain original references and caps. Final bridge200PASS and strict
bridge lintPASS; native adaptive/cold4PASS, modal1PASS (both orders), thermal/static1PASS and
frequency8PASS/1intentional producer entry ignored (49.45s); strict native lintPASS. The producer
entry is exercised by process witnesses and receives no separate unique-test credit. A wrong
frequency filter selected zero and is archived separately, without credit. Main architecture and
133-member language ledgerPASS; exact candidate retains only the two excluded HEAD renderer-size
debts. Formatting/diff and107owned/8239excludedHEAD/98preserved/indexempty auditPASS. All handles
terminal; FEA source hashes are unchanged by the final redundant-reference cleanup in the bridge.
The draft is not retained yet; twelve original scientific failures and all full stage gates remain.


### Modal and frequency compiler contribution (303)

Revision86 requires original compiler peak storage in modal extraction, frequency entry and
frequency field-recovery demands. Their companion planners accept the same opaque borrowed
construction carrier; source scope is validated before deriving numerical facts. Plan-only
callers reconstruct bounded original compilation and compare the complete physical plan.
Equivalent original inputs independently compiled yield identical canonical requests. Serialized
storage counts cannot supply construction authority; missing or zero fields reject cold decoding
or planning, and positive rehashed understatements reject fresh native rederivation.

Modal bridge preparation carries its original bounded construction through resource planning.
Frequency fan-out carries the original construction through a projection that derives shared
numerical facts once. Native modal, entry and recovery workers reserve original encoded/decoded
input retention and fixed worker storage before compilation; original construction is released
after resource validation and before physical execution. No candidate report stays live as
unaccounted solver scratch. Numerical frequency projection bodies match HEAD after module-path
and storage-field normalization; derivation and admission are separate child modules.

Monolithic tasks retain the larger compiler/numerical phase, plus original inputs and worker
storage. Consequently a larger residual-vector workspace does not always increase the whole-task
peak. The original modal resource fixture now checks strictly larger solver workspace and phase
storage, nondecreasing task storage, and the exact compiler-dominated peak when requests coincide.
Its meshes, numerical references, ceilings and solver policies are unchanged. This is a correction
to resource composition evidence, not a relaxed continuum criterion.

Final resource tests175PASS14.97s (build26.73s) and strict FEA lib/test lintPASS24.29s.
Frequency49PASS17.79s; original damped-modal5PASS393.22s. Combined bridge/native strict lintPASS
3m59s; bridge200PASS5.16s; native modal1PASS1.85s (both element orders), frequency9PASS50.18s plus
one intentionally ignored producer entry, and adaptive/cold4PASS1.79s. Process witnesses exercise
producer termination, original cold fields/evidence, invisible precommit loss and retry. New native
frequency tests verify positive understated, recomputed resource plans at solve and recovery for
both Tet4/Tet10. Repeated producer/test selections receive no independent aggregate credit.
Only two Rust test files changed after numerical/native verification; all production hashes
are unchanged and the corrected tests/lint are separately verified. Documentation is updated afterward. Initial compile errors
and174PASS/1FAIL resource assertion remain archived without pass credit.

Combined291–303 draft120paths remains uncommitted. Scope format/diff and exact candidate audits
preserve98 unrelated tracked edits and an empty index; the candidate retains only the two excluded
HEAD renderer-size debts. Requested storage is not RSS, a full-task bound or actual allocation.
Allocated execution context, remaining compiler families, later owner phases, full adaptation,
all twelve original scientific failures and the blocking release remain open. No stage
promotion. Evidence prefix `/private/tmp/runmat-fea-modalcompiler303-`.

## Scheduler-reserved analysis handoff (304)

The existing scheduling owner reserves the original task request before dispatch. Local
process handoff now carries that resource request beside the unchanged original
`ProgramExecutionRequest` in the shared artifact contract. Remote worker dispatch obtains
it from the original validated scheduling assignment. Program schema, canonical program
identity, physical identities, meshes, controls and authored ceilings are unchanged. Cold
metadata validation proves shape, not independent scheduler ownership.

The shared analysis boundary requires a current governing resource plan and compares its
request against the carried reservation before entering the numerical kernel. CPU, memory,
scratch, duration, artifact, egress, relay, capabilities and accelerator class/count/memory
are checked independently. Spare capacity in one dimension cannot replace another. Duplicate
accelerator classes reject at the shared request contract, preventing separate capacity
checks from accepting claims whose combined reservation exceeds a class's inventory.
The opaque reserved invocation is constructed only by shared admission. Explicit standalone
execution remains available; scheduled execution does not silently fall back to it.

This admission follows immutable input closure. It does not yet establish bounded pre-import
decoding, whole-task allocation, allocator overhead or RSS. Source-only and legacy phase
routes without a canonical governing plan reject rather than infer authority from a large
reservation. Their canonical phase planning remains required work. Meshing stdio uses the
same handoff format, but this slice does not reconcile meshing numerical allocation.
Remote plumbing compiles; the required origin recheck encountered GitHub DNS failure, so
no current platform convergence or remote/distributed/GPU promotion is claimed.

Final exact-candidate shared execution41PASS, artifact38PASS, bridge202PASS, generic runner
4PASS; shared/artifact and combined bridge/native strict lib/test lint pass. Native worker
1PASS, modal1PASS (both orders), frequency9PASS plus one intentional producer-only ignore.
Original scheduled J2 process lifecycle passes Tet4 and Tet10: seven phase closures per
order, original accepted histories/ordinals, cold authentication after producer lifetime,
receipts and runtime publication. Meshing lifecycle protocol
passes cancellation, deadline, exit/retry/checkpoint and spawn-failure witnesses. These process
witnesses preserve original limits and do not replace the outstanding continuum references.

The oversized artifact request module is split into its production and test owners. Its
original production body and test body are byte-identical to HEAD; no schema or validation
semantics changed. Local request serialization now uses the shared bounded JSON codec before
framing. All owned formatting/diff checks pass; main architecture passes, while the exact
candidate retains the same two excluded HEAD renderer-size debts. All98 unrelated tracked
edits remain unchanged and the index remains empty. Combined291–304 draft148paths is
uncommitted; no new build/worktree root or subagent was used. All twelve original scientific
failures, remaining compiler/owner admission, full adaptation and attested release stay open.
Evidence prefix `/private/tmp/runmat-fea-allocation304-`.

## Original input read and decode admission (305)

Scheduled analysis now charges the original program's shared structural profile against its
carried memory reservation before fetching the host descriptor. Host and numerical manifest
reads use the existing `CacheImport::read_verified_bounded` contract; providers that cannot
enforce a bound reject instead of performing an unbounded read followed by a length check.
Canonical metadata is structurally profiled before value-tree verification/typed decoding.
Each numerical manifest's complete payload demand and imported metadata are checked before
its first chunk read. Read occurrences are charged separately even when digests repeat,
matching the original importer’s owned buffers rather than silently deduplicating its history.

The same shared structural scanner admits resource-plan decoding before construction and
all original payload materialization before numerical kernel entry. Missing governing plans
still produce their original specific rejection before unrelated typed payload decoding.
Capacity errors are typed resource diagnostics containing required and reserved byte counts;
overflow and cancellation reject. Source access, manifests, chunk order, hashes, numerical
identities, fields and physical history are unchanged. This is conservative requested-storage
admission. Transport buffers/decoding before this analysis entry, domain validation scratch,
later assembly/solver/output overlap, allocator overhead and measured RSS still need full-task
qualification; no exact heap/RSS claim follows from these estimates.

Native execution now constructs the shared response from the same admitted host used for
execution. It no longer decodes and retains a second copy of that descriptor. The import
owner and unchanged existing resource tests move out of the oversized object root: production
root449lines, import280, admission100, conversion258, resource tests194, serial314. A typed
borrowed import request keeps expectations together; no argument-limit waiver or new cache,
codec, numerical kernel or scheduling authority is introduced. Explicit standalone APIs
retain their original behavior and do not acquire a fabricated scheduler reservation.

Final bridge205PASS2.00s (build5.83s), including three provider-read witnesses: exhausted original
invocation causes zero reads/kernel entries; oversized complete closure reads only its manifest;
positive import preserves repeated chunk occurrences and complete prepared canonical identity.
Combined bridge/native strict lib/test lintPASS1m54s. Native modal1PASS1.74s (both orders),
frequency9PASS42.42s plus one intentional producer-only ignore. Original scheduled J2 process
PASS Tet4/Tet10 through all seven phases, cold original history/receipts and runtime publication
(build19.93s), unchanged caps. No overlapping aggregate credit.

Initial bridge201PASS/1FAIL identified rejection ordering and remains archived; missing test
import and excessive helper arguments were corrected, then one item-order lint failure was
fixed without suppression. Final owned format/diff and preservation audits pass. Main
architecture passes; exact candidate retains only the two excluded HEAD renderer-size debts.
Combined291–305 draft155paths is uncommitted; all98 unrelated tracked edits and empty index
are preserved. All twelve scientific failures, missing phase plans (including static field
recovery), remaining compiler/owner admission, full adaptation and attested release remain open.
Evidence prefix `/private/tmp/runmat-fea-input305-`.

## Original static field-recovery admission (306)

Revision87 adds canonical field-only admission for original static structural results. The
resource planner binds the original recovery request/result, mesh or explicit topology,
canonical constraint bytes and actual bounded original compiler peak. A shared inventory
selects exactly the fields retained by move-only static recovery, plus canonical constraint
reactions. Output catalog and partition profiles cover every published original field, even
when observations request only displacement. Fields are neither discarded nor reconstructed
through equilibrium to fit the reservation. The existing shared host-phase accounting carries
input decoding, compiler/field work, output construction and worker allowances under the
unchanged original authored accuracy ceiling and wall limit. Compiler storage is released
before field publication; this phase carries no artificial structural solve/operator demand.

The bridge prepares the immutable governing plan alongside the original invocation/result and
optional mesh. Native execution and cold import independently reconstruct original constraints
and the full plan. Missing plans, rehashed positive compiler/workspace understatements and
foreign constraint digests reject. The resource contract's closed enums move unchanged into
a focused95-line owner; contract root514, preparation232, resource199, native structural153 and
native recovery221. No warning suppression, compatibility fallback, extra cache/scheduler or
module-size waiver. Original J2 recovery helpers remain unchanged.

Final isolated verification:175 resource regressions pass(10.91s),205 bridge tests pass(2.06s),
FEA/bridge strict lint passes(48.96s), native strict lint passes(1m11s). Native full linear pipeline
passes(1.03s), including cold plan rederivation and positive rehashed/missing admission rejection;
thermal-source recovery passes for Tet4/Tet10(3.41s, one both-order witness). Real scheduled native
solid, beam-only and mixed solid/beam/shell witnesses all pass(build7.32s). After each producing
session is dropped, the filesystem store is reopened under the existing retained-object-source
lifetime guard; every original published field and its source identity compares equal. No
producer session, decoded model or reconstructed numerical state is retained by that guard.
Initial cold witness failed when session teardown released its directory; the corrected witness
uses the intended shared lifetime contract. Initial stale duplicate-accelerator fixture failure
is archived and now checks invalid construction at the earlier shared gate. Copy-clone lint and
missing import/API errors were corrected without waivers; none earns terminal pass credit.

This is requested component storage and execution evidence, not measured full-task heap/RSS,
independent physics convergence, atomic adaptive-root publication or attested release. All12
scientific failures and full Stages1–12/Definition-of-Done gates remain open. Main architecture
passes; the isolated candidate retains only the two excluded HEAD renderer-size debts. Combined
291–306 draft171paths remains uncommitted with98 unrelated tracked changes preserved; no Git
mutation or retry of the prior approval timeout. Existing candidate73/request50/checkpoint49
roots reused. Evidence prefix `/private/tmp/runmat-fea-staticrecovery306-`.

Post-verification cleanup, with all test handles terminal and exclusive Cargo locks:99 obsolete
incremental configurations removed,4.73GiB immediate free space recovered.443 current configurations,
175 executable variants, all libraries/dependencies/sources/evidence retained. Executable audit found
no obsolete duplicates. This is distinct from the earlier post305 cleanup; no inferred cumulative total.
