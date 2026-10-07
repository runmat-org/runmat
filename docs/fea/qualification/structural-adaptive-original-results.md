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
copy/evaluation construction before payload copies; cancellation reaches the
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
closed adaptive output/root, atomic semantic publication, durable retry/replay,
producer termination and remote execution are still pending. The affine witness
also does not resolve the original cantilever convergence gate, qualify nonlinear
history transfer or close Stage 11.
