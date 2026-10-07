# Structural adaptation from original solved results

This is an experimental in-process evidence and acceptance path. Stage 11 and the
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
remain unchanged. Explicit-surface coupling borrows its original coordinate map;
this slice does not account for constructing that separate topology owner's geometry.
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
