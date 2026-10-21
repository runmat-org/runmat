# Small-strain J2 solid qualification — implementation in progress

Status: **unqualified**. No Stage 6 exit, execution shape, release or qualification gate is
closed by this draft. The original scalar nonlinear pipeline has been replaced in source;
public-route compilation succeeds, but independent continuum gates have failed. A new kernel
or passing adjacent linear tests is not whole-route qualification.

## Current lifecycle and assembly evidence (2026-10-08)

Status remains **unqualified**. The exact candidate from iteration 335 passes 13 native
parent continuation tests, with two ignored subprocess entrypoints (569.62 seconds).
Invoked child helpers do not add independent qualification credit. The suite covers
original Tet4/Tet10 homogeneous and layered histories, fields, schedule/control state,
cold replay, producer/publisher termination, corruption, order, budget, cancellation,
and accuracy publication gates. Native/runtime library and test compilation passes;
strict native library/test lint passes. This is not a full native-suite result.

Nine element, four Newton, nine driver, 175 resource and 207 bridge tests pass, along
with strict FEA/bridge library/test lint. A subsequent test-only oracle extension in
iteration 337 passes the same nine element tests (2.43 seconds) and strict FEA
library/test lint. Native evidence predates that test-only extension; production code
is unchanged. These groups overlap and are not a single aggregate corpus.

The complete corpus from iteration 330 remains **1,871 passed / 19 failed of 1,890**:
twelve original scientific failures and seven cancellation/resource regressions.
Scoped correction checks pass the seven regressions; there is no corrected whole-corpus
pass. Eight original J2 gates remain failed. The fresh 15/125/512-point trajectory rerun
still fails, with every reported field/history/energy/moment error matching iteration 330.
See the progress map for exact names and the ledger for timings and source manifests.
Earlier CAD, broad native and runtime results are historical evidence. Full accuracy,
measured resources, backend/execution envelopes, Stage 6, Definition of Done and release
remain open.


## Current admitted block implementation (2026-10-02)

The local CPU pipeline now has an explicit incomplete-block output, distinct from
completed public study results. Shared preparation owns formulation admission,
resource planning, source realization and physical assembly for both paths.
Controller checkpoints retain original domain/schedule, pending nominal target,
complete accepted point history and cumulative accepted/rejected/Newton/backtrack/
evaluation work. Canonical transport requires an external expected content digest;
it grants integrity only, not scheduler or semantic restart authority. Execution
revalidates the original controller scope and accepted physical equilibrium.

One validated volume recovery owns both fresh and resumed output. Resumed initial
plastic strain comes from original point history. The full-solve wrapper rejects
an unfinished schedule when its global accepted cap is exhausted. Failed cutbacks
retain their attempted work and original pending target across block boundaries.
Planning and execution share checked work bounds; resource revision62 explicitly
charges block-prefix replay. Original authored ceilings and numerical rules remain.

Driver8/8, selected block7/7 (five driver plus two admitted-pipeline tests),
preflight2/2, payload1/1 and shared resource167/167 pass; strict FEA lib/tests lint
passes. Both orders match uninterrupted original point states and physical fields,
including nonuniform histories. Independent homogeneous hardening checks pass.
This does not establish independent nonuniform continuum accuracy: selected broad
nonlinear regression33PASS/8FAIL retains the seven original scientific failures
plus the existing richer-rule fresh-trajectory failure.

Native task transport, real producer termination/cold continuation, source-bound
numerical receipts, shared accuracy and original-state publication now pass focused
both-order witnesses. The authored study/runtime coordinator, retained-prefix admission,
standalone retention and whole-study qualification remain open. Driver-owned earlier
outputs are not silently covered by one worker's resource plan.

### Original strain and all authored states (2026-10-03)

Assembly's original volume-mean total engineering strain is now retained alongside
stress and plastic history at every accepted increment. The public terminal result
exposes `structural.strain`. Event-source v4 binds it; resource revision66 includes
its numeric history and vector inventory. Artifact validation checks DOF/element/
six-component tensor shapes, including rejection after rehashing malformed strain.

Typed publication from the original terminal block covers all three declarations:
`StructuralPrimary`, `StructuralStressStrain`, and `MaterialHistory`. Each retains
original displacement, strain/stress evidence, point history and controller; it does
not rerun equilibrium or manufacture point history from averages. Fourteen real
native producer exits precede cold imports across both element orders. Independent
homogeneous references check all six total-strain components as well as stress,
displacement, plastic strain and alpha. Missing mandatory mesh evidence remains
Indeterminate and blocks publication; no continuum promotion follows from this witness.

Authored product planning revalidates compiled mesh/material identity and preserves
the original model, declarations, schedule and caps. Matching relative nonlinear
closure criteria may tighten stopping controls. Initial and continuation requests
now derive their exact original ordinals and authenticated predecessor references.

## Intended bounded envelope and ownership

Canonical Tet4/Tet10 isotropic, small-strain quasistatic solids; associative J2 yield with
constant initial yield stress and linear isotropic hardening. Fixed dead loads and complete
zero-offset affine equality constraints. Each element consumes its own compiled material;
there is no first-material/global constitutive option. The authored material law carries
dimensional `yield_stress` and `isotropic_hardening_modulus` and an explicit closed law name.
Old yield-strain/saturation proxy inputs are rejected, not converted.

The constitutive update is a pure trial from committed quadrature history. Engineering
strain order is xx, yy, zz, yz, xz, xy, with engineering shear and physical shear stress.
The physical assembly computes internal force Bᵀσ and sparse tangent BᵀC_algB using actual
physical quadrature weights. Newton solves TᵀK_algT δq = −Tᵀ(f_int − λf_ext).
Line-search candidates all start from the same committed point history. Only physical
equilibrium can accept displacement/history; failed increments and exhausted cutbacks do
not mutate the caller's accepted state or publish partial success.

Finite strain, contact/friction, follower loads, thermal/coupled laws, explicit/mixed
elements and connectors, nonzero initial lifting, GPU/distributed execution and prestressed
downstream use require separately implemented and verified envelopes. They are not supplied
by this draft. Perfect plasticity is supported by the local kernel; force-controlled global
collapse must reject on loss of equilibrium/tangent admissibility, not fabricate convergence.
Authored checkpoint/completed-state restart requests cannot silently initialize fresh history.
Boundary operators are rejected in this envelope rather than omitted from forcing; supported
mechanical pressure/traction must use typed excitation actions.

## Independent evidence chain

The implementation now has a mesh-bound `CanonicalSolidIntegrationPlan`: exact
ordered element/point/node inventory and exact reference-rule identity, using the
shared canonical codec and geometry/rule evaluators. Keast15 remains the unchanged
public solve rule; positive Duffy5/8/12/16 are explicit internal diagnostic plans,
not qualified replacements. Physical basis and original point histories bind the
plan identity. Foreign-rule history is rejected, never interpolated or initialized
silently. All five rules preserve independent affine stress and face-integrated
force for both orders; this does not qualify moving-front integration accuracy.

Successful nonlinear results retain the complete terminal accepted state, including
original ordered plastic strain/alpha at every integration point, physical displacement,
load factor, accumulated dissipation and basis/load/constraint/integration identities.
The driver revalidates terminal equilibrium/history before returning success, and
the numerical content digest binds this carrier. Element averages remain output,
not continuation authority. Same-process typed cold round trips and cross-rule
rejection are tested; shared semantic publication, governed resource admission and
worker/process-separated restart are still outstanding. The carrier alone grants
none of those capabilities.

The public `FeaNonlinearRunResult` now also retains that same opaque carrier by
move, rather than dropping it while constructing projected fields. Both-order
public completion tests check exact ordered inventory/integration identity, original
displacement, cold serialization followed by physical scope/equilibrium validation,
and rejection of a tampered point alpha. Tet10's nonuniform original alpha differs
within an element and cannot be replaced by the element mean. Actual complete public
result wire profiling fits the requested bound in the explicit eight-accepted-step
test envelope. No old result decoder may fabricate absent point history.

Integration-plan retained/derivation storage estimates reuse shared mesh-validation
and canonical-codec profiling without physical point construction. They count
requested storage, not allocator overhead, measured peak RSS or whole nonlinear
task demand. The direct CPU pipeline now derives a distinct nonlinear resource
plan before excitation realization and physical assembly. Its additional persistent
storage includes prepared basis, original point histories, trial/validation evidence,
planned sparse tangent support/copies, authority construction, retained accepted
snapshots and content-identity construction. The accepted cap, not the nominal
increment count, governs retained output. Numeric content uses allocation-free
shared codec profiling; map/point metadata keeps its conservative allowance.
Both-order actual complete driver content fits the projected wire/residency witness.
Newton now consumes a solver admission bound for the actual reduced tangent system,
including models whose authored constraints use direct elimination. Each accepted
increment is consumed immediately rather than retaining a prefix of full tangents
and point evidence; cutbacks still reach the original scheduled target and preserve
global accepted-step limits. Borrowed assembly/options remove extra owned mesh copies.

This is authored-ceiling and inner tangent-work admission, not independently
scheduled worker allocation, measured process memory or complete resource
qualification. Planning construction itself, portable task rederivation, usage
receipts, semantic-state publication and process restart remain open. Unsupported
restart/boundary requests reject before assembly. Canonical body-force volume
integration is now correctly attributed to nonlinear solid realization; the retired
equal-node source cannot supply that evidence.

Latest original public matrix: 4/7 pass, 3 physics failures. Tet10 one-step cells1/2
passes unchanged1e-7 targets, with errors around1e-14–1e-13. Tet10 eight-step cells1
plastic-field error2.93625e-5 fails1e-7. Tet4 cells1→2 stress/reaction-related errors
do not satisfy the original refinement assertions on either schedule. Earlier
source-axis, reduced-tangent dimension and output-envelope rejection failures have
been corrected; no scientific threshold, authored8GiB ceiling or4096 accepted cap
was changed. These original scientific failures prevent whole-route qualification.

1. Constitutive elasticity/hydrostatic/pure-shear/uniaxial analytic responses, plastic
   incompressibility, yield consistency, unload/reload/reverse yield, stored elastic and
   hardening energy, nonnegative irreversible work, and fixed-committed-history finite
   differences for all six engineering tangent columns.
2. Canonical both-order affine strain patches with independently integrated face tractions,
   actual physical energy and sparse internal-force tangent finite differences. Independent
   disconnected bodies with different yield/hardening laws and order-reversal invariance
   test material ownership, not heterogeneous interface qualification.
3. Actual Newton equilibrium for a uniformly stressed unit tetrahedral bar. E=1200 Pa,
   ν=.25, σ_y0=5 Pa, H=30 Pa, imposed end stress=5.6 Pa. Independent uniaxial-stress
   solution: α=(σ−σ_y0)/H; ε_xx=σ/E+α; ε_yy=ε_zz=−νσ/E−α/2. Physical volume=1/6 m³;
   total irreversible work=σ_y0 α/6. Component constraints on x=0/y=0/z=0 remove rigid
   modes without suppressing the compatible lateral contraction. Displacement, point α
   and work gates are fixed at absolute 2e−10 in their respective SI quantities.
4. Actual accepted-prefix cold typed serialization followed by the same physical Newton
   advance, compared against uninterrupted analytic response. Negative tests cover foreign
   load scope, non-equilibrated load factor, overflow, changed accumulated work and failed
   trial/cutback rollback. This unit cold reopen is not worker/process/durable publication
   or a general shared state-transition/restart qualification.
5. Frozen independent nonuniform continuum and public-route fixtures cover mesh/load-step
   convergence, exported fields and physical constraint reactions. Their numerical
   failures are recorded below. Authored observation/state publication, admitted resources and
   real shared execution/restart evidence are still required before a gate decision.

## Next nonuniform continuum reference

Use a unit cube with compatible u_x=a x+b x²/2 and all u_y/u_z constrained to zero.
In the final fully yielded monotonic state e=a+b x, G=E/[2(1+ν)], K=E/[3(1−2ν)],
α=(2G e−σ_y0)/(3G+H), and

σ_xx = E_t e + σ_0,
E_t = K + 4GH/[3(3G+H)],
σ_0 = 2G σ_y0/(3G+H).

The exact dead body force is f_x=−E_t b and right-face traction is σ_xx(1);
left-face u_x=0. Lateral component constraints supply the physical lateral reactions;
this is a uniaxial-strain manufactured solution, not free uniaxial stress. Tet10 represents
the displacement exactly on affine cells; Tet4 must demonstrate mesh convergence.
References and thresholds must be fixed before running this gate, and energy/history
must come from accepted quadrature states, not displacement-based plastic proxies.

The predefined primitive matrix uses cells1/2/4 and one/eight requested increments for each
order. Tet10 final displacement, stress, alpha, separate energies and lateral reaction
moments must each have relative error below1e−7 on both schedules. Pointwise axial strain
must advance monotonically at every accepted subincrement, with only strain-scaled floating
point roundoff. Tet4 displacement reductions exceed3 and stress/alpha reductions1.5;
finest respective relative bounds are.05/.15/.15. Energy reductions exceed2 unless both
errors are below1e−10 (an exact integrated reference), with finest energy bound.08 and
lateral moment bound.15. All cases require axial reaction balance below1e−8 relative.
Public-route tests use both orders, cells1/2 and both schedules; Tet10 remains exact1e−7,
while predefined Tet4 coarse/fine exported-field bounds are.35/.2 with reduction above1.5
or the same exact-reference branch. Failed gates remain failed; none is a qualification gate
by itself.

## Verification record

2026-09-30: closed law, kernel, physical assembly, reduced Newton and independent unit
tests are drafted. Core law tests pass 2/2. Initial FEA/language builds reject unused draft
code with warnings denied; no warnings are suppressed. Draft mechanics are temporarily
test-linked while their independent tests compile and the public pipeline is replaced.
Newton tests pass2/2, kernel11/11 and the same compiled binary's assembly tests5/5.
Language tests remain unverified after concurrent schema/manifest and incomplete-lock changes.
07:11 UTC: public-pipeline/driver cutover, old scalar/global carrier deletion and dimensional
runtime constructors are implemented. New result stress and force are material/equilibrium
owned, and reactions use actual internal minus scaled applied force. CPU,
canonical mesh and dead-load requirements fail closed; synthetic/contact/coupled/GPU and
nonzero initial-history routes cannot borrow this identity. A first library build rejects
unused draft evidence under warnings denied; production recovery now validates that evidence
without suppressions. An ambiguous test constructor was corrected. New checks are queued;
no coherent implementation commit or whole-route qualification is claimed. The old incomplete
concurrent lock boundary has cleared for locked/offline metadata without editing its owner’s
manifest. Runtime fixture rejection migration remains in progress.

07:43 UTC: controlled non-test library check passes; core and language dimensional material
tests pass2/2 each. Final nonlinear continuum/public-pipeline, runtime and strict all-target
verification remain live on the shared build lock. Subsequent source review removes external
arbitrary-summary solve entry, adds actual nodal reactions and empty unsupported contact/coupled
channels, and checks cold reporting-field identity. The new runtime fixture executes genuine
both-order canonical solids and cold-reopens stored reporting history; it is not constitutive
restart or worker/process qualification. Concurrent schema feature compile failures are recorded
separately from physics; no final numerical, qualification or release gate is claimed.

Primary method context: [MOOSE isotropic plasticity stress update](https://mooseframework.inl.gov/source/materials/IsotropicPlasticityStressUpdate.html)
and [radial return stress update](https://mooseframework.inl.gov/source/materials/RadialReturnStressUpdate.html).
These describe the method; they are not a substitute numerical oracle for the analytic
stress, displacement, energy and equilibrium tests above.

## 08:51 UTC verification and gate decision

Clean isolated FEA build completed in39.14s, avoiding the unrelated shared build queue.
The first nonlinear run18261 passed27 and failed7: five invalid fixture/expected-rejection
assumptions and two numerical cases. Vector signals now explicitly name the global frame;
internal driver fixtures author their actual support components; coupled execution asserts
the earlier truthful unsupported-family rejection. No existing validation was weakened.
The corrected run70718 passes32 and fails2. Splitting order/schedule tests, without changing
any numerical predicate, exposes the full result in43902:33pass/5fail,198.42s.

- Tet4 continuum displacement errors at cells1/2/4 are.0955202/.0394160/.0122926;
  the first refinement ratio2.423 fails the frozen >3 gate.
- Both public Tet4 schedules fail averaged-field refinement. For one increment,
  coarse/fine von Mises errors.00170336/.00481781 and alpha.0189843/.0531759 grow;
  cancellation in element averages is not continuum convergence evidence.
- One-increment Tet10 public cells1/2 passes all exported-field/reaction/cold reporting
  assertions near1e−13. Primitive cells1/2/4 one-increment fields/energies/reactions are
  also near roundoff, but the combined primitive test fails the eight-step branch.
- Eight-increment Tet10 primitive displacement errors9.27309e−6/3.28014e−6/8.06291e−7
  converge but fail frozen exact1e−7 gates. Corresponding alpha errors are
  1.44109e−5/6.07674e−6/1.64026e−6. The public coarse eight-step case also fails.
- Fresh kernel checks pass11/11; physical thermal regressions pass45/45. These do not
  qualify nonlinear continuum behavior. Runtime integration76326 fails compilation on
  an unused import in concurrently owned source/query.rs before any nonlinear execution.

Independent path review identifies a false inference, not a justified operator fix:
monotone axial strain alone does not establish a proportional deviatoric FE trajectory.
With C=K+4G/3, T=E_t, O=σ_0, and final S(x)=S_0+S′x, the exact scaled-load history is

```
r(λ) = clamp((C σ_y0/(2G λ) − S_0)/S′, 0, 1), r(0)=1
e(x,λ) = λ S(x)/C                       when x ≤ r
         (λ S(x) − O)/T                 when x > r
alpha = 0                              when x ≤ r
        (2G e − σ_y0)/(3G+H)            when x > r
u(x,λ) = integral_0^x e(s,λ) ds
```

First yield isλ=.209568886861314; full yieldλ=.384092809364548. The eight-step
trajectory crosses non-mesh-aligned fronts atλ=.25/.375, not globally quadratic fields.
Finite-mesh intermediate shear/rotating plastic directions can persist in history even
when final continuum displacement is polynomial. The current exact eight-step gates remain
**failed**, not relaxed or silently replaced. A subsequent history qualification must use
the actual independent moving-front trajectory and preregistered spatial/load-step evidence.
No Stage6, qualification, worker/restart or release promotion; source remains an uncommitted draft.

## Moving-front history discriminator (2026-09-30)

Added independent piecewise displacement/strain/stress/alpha and closed, front-split
elastic/hardening/dissipation/lateral-moment integrals. Self-checks use equilibrium,
J2 consistency, compatibility, front continuity and the final analytic limit. A first
run caught an oracle endpoint bug: a clamped front at x=0 does not make that endpoint
elastic after full yield. The corrected oracle classifies by axial stress. No production
operator, existing reference bound, solver setting or failed gate was changed.

The local constitutive discriminator compares one/eight/64 increments at seven fixed
locations against all six analytic plastic-strain and stress components, separate energies,
and accumulated incremental dissipation. The corrected oracle and local discriminator pass.

For cells=4 Tet10, the independent schedule places fronts at x=1,.75,.5,.25,0,
then advances to full load. Each accepted front is an element-layer interface: the
piecewise-quadratic displacement is representable, and no unaligned accepted cutback is
permitted. This exact-path test passes: final displacement/stress/alpha errors are
2.30e-13/4.12e-13/4.82e-13; energies and lateral reaction moment are below3.31e-13.
This supports representable-history equilibrium, not uniform-schedule exactness or an
envelope limited to specially selected load factors.

The separate preregistered cells1/2/4, uniform-eight-increment matrix preserves the
previous fine stepped bounds across the entire history (u .02, stress .05, alpha .08,
energies and lateral moment .08), and requires successive refinement or near-roundoff
error. Alpha/energy normalization uses the full-load reference inventory to remain defined
in the elastic interval; displacement/stress use current-load references. Both new uniform
gates fail: Tet4 stress at factor.375 is.0543288>.05; Tet10 alpha errors there are
.000190539/.000116414/.000164692, not monotonically decreasing. The pointwise norm uses
native quadrature; moving-front sampling and true spatial-history error still need to be
distinguished by independent front-split error integration. Neither failure proves a local
constitutive defect or licenses weakening the gates. The original five failures stay failed.

Latest evidence strengthening compares every accepted subincrement, checks point/history
cardinality and compares the solver's accumulated-work carrier with weighted material history
before using that carrier for the independent energy comparison. Final rerun63010 terminates
with3 passes/2 unchanged uniform-history failures (167.86s); strict FEA library/test Clippy
passes. No coherent nonlinear implementation commit,
worker/process restart, authored publication or Stage6/release promotion is claimed.

## Independent output-field integration and decomposition

The actual element-average stress/alpha output is a piecewise-constant field. Its error
can be integrated independently without inventing constitutive histories between points:
clip each straight tetrahedron at the analytic front, then use exact affine-square simplex
moments on each side. Cube integrals are independently cross-checked with piecewise 1D
Simpson integration, including loads whose fronts coincide with mesh planes. This tests
integration, not a second constitutive oracle. Curved cells are outside this test helper.

The implementation separately checks the orthogonal decomposition
`output error² = mean-value bias² + reference variation within cells`.
The projection term must not be mistaken for constitutive accuracy. At factor.375, Tet10
alpha output errors for cells1/2/4 are.05999794/.02993450/.01491266, but the independent
mean-bias errors are6.60380e-5/9.39196e-5/8.17939e-5. Thus output convergence does **not**
erase the unresolved history error or establish that it was merely quadrature sampling.
Tet4's cells4 stress error remains.05431107 under exact front-split output integration,
close to the point metric.05432876 and still above.05. The coarse-grid failure is real.

Run53089 passes4/fails2 (196.62s): oracle, local radial histories, front-split integration
and aligned Tet10 pass; original two uniform-history predicates fail. Both orders' new
cell-output monotone-refinement assertions execute successfully before those failures.
The preceding run91894 failed3/6 because nearly coincident clipping planes generated a
collapsed cut face; coordinate coincidence is now resolved with a32-epsilon unit-fixture
arithmetic bound, and exact integral tests cover the aligned front loads. No physics
threshold or operator changed. Strict latest FEA library/test Clippy28754 passes20.77s
after replacing a flagged indexed loop with an iterator; formatting and whitespace pass.

Additional cells4/8 Tet4 history qualification12535 passes (252.81s) with the same trajectory
bounds and original final-state rate targets (>3 displacement, >1.5 stress/alpha). Cells8 final
u/stress/alpha errors are.00322487/.02301529/.02776541; cells4→8 reductions are3.733/1.919/1.917.
At the previously failing factor.375, cells8 stress error is.02792124, below the unchanged.05.
This supplies finer-mesh/asymptotic evidence without establishing any mandatory execution,
resource or accuracy-report acceptance. This is additional
resolution evidence, not a replacement for the failed coarse-grid cases. No qualification,
release or whole nonlinear execution qualification is inferred from these diagnostics.

## Captured accepted-path discriminator

The test-only `moving_front/captured_path` diagnostic retains every original accepted
increment, including cutbacks, on uniform cells1/2/4 Tet4/Tet10. Original point IDs,
all six plastic-strain components, alpha, stress and incremental/accumulated
dissipation replay exactly from the preceding accepted history. This bookkeeping
matrix passes1/1 (199.25s); it is not independent constitutive qualification.
A separate scalar axial counterfactual driven by captured axial strains quantifies
the effect of induced FE shear without replacing the real state or continuum oracle.
Original uniform-history failure predicates remain unchanged.

Frozen-force reintegration uses125/512 positive Duffy points, each evolving its own
history from zero through the captured displacement trajectory. It neither
interpolates old point history nor reruns equilibrium. Direct tensor virtual work
compares the reduced force against original integration and between richer rules.
Tet4 and elastic Tet10 must agree within1e-8, registered before execution; nonlinear
deltas remain diagnostics, not assertions of converged quadrature or
qualified accuracy. Full enriched matrix passes1/1 (220.56s), all6 mesh/order cases and
all accepted subincrements, including the cells2 Tet10 factor.4375 cutback.

The expanded125/512/1728/4096-point diagnostic matrix passes1/1 (428.52s).
It captures each original equilibrium trajectory once and streams one new point's
complete history at a time. Point states are not retained across the entire refined
mesh. All original6 mesh/order cases and accepted cutbacks remain covered. This
is diagnostic execution success, not satisfied independent integration accuracy.

At fine Tet10 cells4/factor.375, normalized original-rule differences for125/512/
1728/4096 points are3.31649e-4/3.32947e-4/3.30940e-4/3.29989e-4. Successive
changes are2.30709e-5/1.20611e-5/1.24194e-6. The original-rule difference
persists while refinement differences shrink. At factor.25 the4096-point difference
is4.74039e-4 and last refinement change3.50697e-6. Neither last change meets
the original1e-9 equilibrium target or supplies a certified integration bound.
At factor1 the original difference is7.83780e-9 with last change3.70461e-11.
Tet4 remains near working-precision agreement throughout all four rules.

Decision: exact history bookkeeping is established, and force integration is a
material contributor during front crossing. The nonmonotonic cells1/factor.375
original-delta magnitudes and remaining refinement changes preclude calling a
fixed high-order rule exact or claiming all history failures explained. No default
rule or original threshold was changed. Next combine an explicit integration plan,
exact point-resource demand and complete accepted-history semantic carrier before
qualifying any new rule on fresh equilibrium trajectories. A rule change cannot
continue old point history without separately qualified state transfer.
Strict expanded-code FEA library/test lint passes19.27s; scoped formatting and
whitespace pass. Original Tet4 coarse/uniform Tet10 gates remain unresolved.
At Tet10 cells2/factor.375, original-to125/512 normalized force differences are
6.89061e-4/6.10714e-4 and richer-rule difference1.00285e-4, while the original
accepted residual passes1e-9. This excludes treating that residual as independent
physical integration accuracy, but does not establish an exact richer-rule answer.
At Tet10 cells4/factor.375, the two force differences are3.31649e-4/3.32947e-4
with richer delta2.30709e-5; at factor1 they fall to8.25783e-9/7.50308e-9
with richer delta1.15569e-9. Exact Tet4 checks pass throughout near1e-14.
Further integration convergence must precede an identity/resource-bound operator
policy change. Strict FEA library/test lint passes (40.64s), formatting/whitespace
pass. These diagnostics do not close the original spatial/history gates.

## Fresh-rule trajectory and trace discriminator

Portable block contracts now bind the mesh-separated original workload, predecessor
artifact/controller digests, output ordinals and complete original point state.
Both-order cold artifact tests reproduce the original terminal history and reject
altered lineage and inventories. This is integrity and trajectory-parity evidence;
native worker admission now authenticates and rederives actual source/decode/output
authority before the original numerical pipeline. Both-order native kernel tasks
resume a yielded homogeneous checkpoint and meet independent stress, displacement,
six-component plastic-strain and alpha references. Rehashed understated resource
plans reject; all168 shared resource tests pass. These are task-boundary tests;
generic program/host execution and cold reopened-store import also pass. Separate
producer termination and receipts now pass: four actual native producer processes
run the original generic programs and exit before both-order cold history import.
Accepted residual receipts bind original block controls, point inventory, work and
lineage under additional resource admission. Shared accuracy and typed MaterialHistory
publication now execute through portable/native tasks and cold imports. Whole-study
qualification remains open. Newton now accepts only when both relative force
residual and the undamped pending correction meet the original tolerance. Correction
is normalized by offset-free current/committed displacement, and uses the original
admitted reduced tangent solver. The last applied correction remains a separate
diagnostic; line-search damping cannot establish increment convergence. Update caps
are unchanged. An additional unapplied verification solve per trial has an explicit
checked work bound, cumulative counter and receipt evidence (resource revision66).
Controller v3 prevents residual-only/incomplete-evidence checkpoints from silently resuming under this
criterion. Event-source v4 includes relative increment and original total-strain evidence. Focused physical and
conditioned-operator tests pass; the latter demonstrates small residual with large
correction and is solver evidence, not a new continuum reference.
Deserialization does not establish
equilibrium or accuracy. The original nonuniform continuum gates remain open.

The MaterialHistory payload retains the original workload and complete terminal block,
including controller, original point ordering, all plastic-strain components, alpha,
displacement and dissipation. Incomplete load schedules reject; generic Fields cannot
represent this state. Worker authenticates exact sources, actual decode facts and
borrowed output profiles, then rederives shared publication admission under original
authored memory/wall ceilings. Rehashed understated decode budgets reject. A real-process
witness runs continuation, accuracy and all three state producers separately: fourteen child
processes exit before cold imports across both orders. Published point state equals the
original carrier exactly; cancellation produces no new block and preserves the predecessor.
The original controller now retains cumulative accepted residual and pending-correction
maxima across all blocks. Terminal receipts therefore include earlier accepted history,
not just the current suffix. Cold driver tests compare these maxima against complete
original-run histories; raw block validation requires current samples to fit them.

Shared accuracy uses retained original nonlinear residual and undamped pending-correction
closure. In the both-order witness this contribution is Satisfied, but mandatory missing
mesh convergence is Indeterminate and publication is BLOCKED. No spatial/load-step/reference
evidence is fabricated. Cold integrity does not re-prove equilibrium, establish field
accuracy, qualify adaptation or authorize a different load schedule. Authored study/runtime
cutover, isolated retention and original continuum qualification remain open.

Fresh zero-history Tet10 cells1/eight-step solves under Keast15, Duffy5 and Duffy8
all fail the original1e-7 final field/energy gate (66.89s). Each plan owns its original
point history from zero; no frozen-displacement replay, interpolation or rule switch
is used. Relative displacement/stress/alpha errors are respectively
9.27309e-6/1.19731e-5/1.44109e-5 (15points),
8.74954e-6/1.13310e-5/1.38074e-5 (125points), and
1.13335e-5/1.36260e-5/1.64854e-5 (512points). Hardening/work errors also remain
above the original target. Thus increasing a fixed rule is not a demonstrated fix,
and the512-point trajectory is not an independently exact reference.

An independent analytic edge-trace check passes for cells1/2/4 at factors.25/.375:
the continuum displacement curvature jumps across the interior moving front by
λ E_t b (1/E_t−1/E_elastic)>0. A single Tet10 quadratic edge trace cannot have both
curvatures. The nodal interpolation discrepancy is4.01132e-5m for cells1/factor.25
and remains2.52715e-6m for cells4/factor.25. At full yield the terminal trace is
polynomial and interpolates near working precision. This proves nonrepresentability
of the intermediate trace, NOT a lower bound on best FE error or terminal history
error, nor an explanation of every failed gate. Exact terminal polynomial shape
alone cannot establish exact representation of an entire path-dependent trajectory.
Original criteria remain unchanged; spatial/history error and its governed accuracy
decision require qualification separately from integration error.

Current original scientific recheck (2026-10-03): eight primitive/public/moving-front
cases finish 1PASS/7FAIL in201.57s. Tet10 public one-step passes; all seven original
failures persist with original meshes, references and acceptance bounds. Public Tet10
eight-step alpha error2.9362500729728705e-5; primitive Tet10 polynomial alpha
1.4410941253038852e-5; moving-front Tet10 increment3 alpha errors
[1.905387342539666e-4,1.1641430261528904e-4,1.6469187666538814e-4].
Missing spatial/history convergence therefore remains a blocking scientific gate,
regardless of exact transport, Newton closure or semantic publication.

## Shared authored coordinator and ordered cold replay

The shared execution coordinator now preserves the original load blocks and drives
accuracy, all three declared states, baseline/terminal resource usage and publication.
It delegates numerical work and terminal phases to existing task and state authorities;
it contains no numerical kernel or scheduler. Cold replay rederives each original task
and imports its saved response/accounting without re-solving. Missing, reordered,
foreign and trailing phases reject. Invalid accounting leaves the awaited phase intact.

Retention admission profiles borrowed metadata and original terminal state, adds actual
owned transport buffer capacities and proposed task demand, and bounds the conservative
sum of retained artifact inventories before copies. Completed block inputs are released.
Content-address equality does not imply shared heap storage. These are component demand
bounds; allocator overhead, process RSS and whole execution qualification remain open.

Both-order native coordinator witnesses execute the original two blocks, accuracy,
three state tasks and publication in real producer processes, then replay after exit.
The current native selection passes9/9 (12.37s), including28 child processes across
the original continuation and coordinator witnesses. Shared bridge tests pass186/186
(1.58s). Mandatory spatial accuracy remains Indeterminate and publication is blocked.
## Durable whole-study closure and original prefix access

The standalone immutable J2 study root now carries exact original compiled inputs,
mesh, controls, ordered phase response/accounting associations and a closed descendant
catalog. Copying only that catalog into a fresh store supports both-order cold replay,
accuracy, all three semantic states and publication. A caller supplies the trusted root
digest, authorization, inventory limits and decode-memory ceiling. Manifest and actual
payload codec profiles precede typed decoding; original authored memory is also checked.
Root construction admits original retained data and pending metadata/transport copies.

Fully rehashed changes to block size, missing/reordered phases, schema and unrelated
descendants reject. Unchanged re-encoding passes. Caller-memory, authorization and missing
child failures reject separately. Original output blocks are available by ordinal through
the original task chain, releasing each earlier decoded block as replay advances. This
imports original fields/history; it performs no numerical re-solve or history reconstruction.
Prefix admission also charges the completed study retained by its caller. An independent
homogeneous hardening reference checks the original0.9 load-factor boundary for both orders.

The inventory checks are shared with the retained CFD root, whose duplicated implementation
was removed. These component budgets do not establish measured process RSS, distributed
execution, scientific accuracy or whole-study storage atomicity. Authored `.fea` dispatch/cutover and exact isolated retention remain outstanding,
alongside the seven original scientific gates.


## Native placement and cold runtime publication

The native adapter now places original coordinator tasks through the existing task pool,
including predecessor dependencies and original admitted resource plans. It owns no
scheduler or numerical kernel. A cold runtime source authenticates the exact whole-study
root, authorization and catalog, then imports original blocks under caller-retention
charges. FEA owns shared recovery of original accepted fields, reactions and constraints;
runtime neither re-solves equilibrium nor derives constitutive history. Direct and study
runtime routes share one field-history transport mapping.

Runtime publication retains the accepted output sequence with global field ordinals,
all three original semantic states, accuracy and resource lineage, and immutable root
catalog. DTO/source, cold replay, accumulated history and recovery demand are admitted
against the original ceiling. The public adapter remains CPU-only and truthfully
blocks publication when mandatory spatial accuracy is absent. Requested component bounds
do not establish allocator overhead or measured RSS.

The scheduled Tet4/Tet10 witness uses the original homogeneous fixtures, two original
load blocks and seven real task phases. Producers terminate before a fresh-store import;
only the declared catalog is copied. Cold prefix alpha and terminal alpha use independent
homogeneous hardening references; altered DTO metadata and caller-memory exhaustion reject.
Persisted history is compared with the original block trajectories, including accepted
cutbacks rather than assuming requested increment count equals accepted output count.
This fixture witness does not certify authored CAD/meshing or the `.fea` entrypoint.
Whole-study storage atomicity and scientific gates remain open.


## Authored dispatch and original portable final observations

The draft now dispatches one fresh independent direct nonlinear step through the original
J2 workload and shared coordinator. It preserves canonical CAD/mesh/material realization,
CPU admission, original defaults and authored resource limits. Unsupported
restart, dependent/reduced/multiple-step and remote envelopes reject explicitly. The new
example retains both displacement and stress final declarations.

Global accepted-work capacity and live output capacity are separate resource facts.
A task with eight accepted outputs retains that inventory; the original global 4,096
accepted-increment cap still bounds cumulative execution. No memory ceiling is increased.
The regression verifies that changing only the global cap preserves a bounded output
payload while changing workload identity and cumulative work authority.

Final FEA observations use original accepted fields. Shared typed observation contexts
and source-result accuracy binding now admit J2 final fields without a fake linear workload.
Cold metadata tests cover both element orders and the independent homogeneous 5.6 Pa
stress reference; elastic reconstruction from the same terminal strain would give25.6 Pa.
Foreign source results and altered authored declarations reject. Sampled/history evaluation
still needs its original accepted-path carrier and is explicitly unsupported here.

The portable publication catalog is now implemented through separately admitted original
nonlinear field recovery, shared partition/reduction/assembly tasks, root descendants,
ordered replay and persisted runtime observations. The both-order terminated-process
lifecycle passes, including reverse partition completion, duplicate/missing/reordered/
trailing rejection and fully rehashed resource understatements. Shared resource regression
passes169/169. The original authored CAD/native Tet4 case completes cold transport and
publication, but fails unchanged independent history2e-9 and stress2e-7 references:
maximum plastic error1.10028e-8 and stress-component error1.06760e-6. The original Tet10
case also completes cold transport but fails those same limits: maximum alpha error
6.82237e-8 and stress-component error1.39226e-6. The whole authored witness is terminal
FAIL; neither completed transport establishes physics qualification. Numerical defaults, meshes,
rule, update caps and resource limits are unchanged. These transport results do not close
the seven original scientific gates or authorize publication.

Original recovery now also rederives the analysis-owned physical basis and checks terminal
scope, constraints, equilibrium, replayed constitutive state and accumulated dissipation.
Saved terminal force, stress, total strain, plastic strain, alpha and von Mises projections
must agree with original physical-state projections within a roundoff-only comparison,
independent of the Newton stopping tolerance. Re-evaluation does not advance equilibrium
or reconstruct elastic stress. Both-order tests reject all six independently rehashed
projection substitutions; block integrity alone is deliberately insufficient. Intermediate
history is authenticated original producer data, not independently re-solved physics.
Latest native J2 matrix passes11/11 (54.01s); bridge passes186/186 (1.92s). Requested
workspace/transport accounting is not measured RSS; whole release remains open.

## Original forcing work discriminator

A new independent source check integrates constant body and right-face traction work
analytically over the original unit cube. It checks every representable scalar monomial
through degree one for Tet4 and degree two for Tet10 against the assembled physical load,
before constraint elimination, on all original cells1/2/4 meshes. This uses no constitutive
update or equilibrium solution as its reference. The six cases pass within the original
1e-10 load tolerance; maximum absolute discrepancies are8.17e-14–1.39e-13. Strict FEA
library/test lint passes. The check rules out incorrect fixed dead-load projection as the
history-bias explanation; it does not establish moving-front state accuracy or close the
seven preserved scientific gates. No operator, integration rule, mesh or tolerance changed.

## Independent nonuniform material-layer continuation reference

A separate canonical two-cell Tet4/Tet10 reference uses an aligned x=0.5 material interface,
E=1000, nu=0.25, H=30, yield stresses2/3.5, axial traction5.6 and homogeneous zero transverse
displacement. The proportional plane-strain reference derives strain, alpha, all six plastic
components, transverse stress and yield-only dissipation analytically. It does not replay the
constitutive kernel. At the original0.9 block boundary one layer has yielded and the other is
elastic; the original terminal factor1 yields both with distinct histories.

Actual native-program producers terminate before each cold import. Original ordinal10 continuation,
cancellation rejection, cumulative work, restored initial plastic outputs and numerical receipts
are checked. All three declared semantic-state publishers likewise terminate before cold import
and preserve the original complete block/workload. Accuracy retains satisfied Newton evidence,
Indeterminate mandatory mesh evidence and a blocking publication decision. The expanded native
suite passes12/12; scoped strict lint and132-member architecture checks pass. The exact final refresh
passes1/1 in87.94s, checking every accepted displacement/stress/alpha/plastic output at the
original2e-9/2e-7 bounds through both layer yield transitions. Exact native lib/test strict lint,
scoped formatting/whitespace and132-member architecture/crate-boundary checks pass.

These fixture references do not replace the seven failed moving-front science gates, supply
multi-mesh convergence or establish authored CAD/source compiler, whole runtime study persistence,
measured resources or arbitrary-history qualification. J2 remains unqualified.


## Rehashed yielded checkpoint rejection

Both Tet4/Tet10 controller fixtures now reach a genuinely yielded checkpoint before
substitution. Five cases alter engineering shear, accumulated equivalent plastic strain,
erase plastic history and its dissipation together, reorder point identities, or change
only accumulated dissipation. Each recomputes the controller digest and passes cold wire
decoding before direct physical-owner validation rejects it. Controller continuation also
rejects; the untouched original bytes remain identical and resume to the independent
uniaxial alpha0.02 reference. The controller block suite passes6/6 (0.82s).

This is physical-consistency rejection for those substitutions, not provenance authentication
of arbitrary rehashed history, a native nonuniform substitution witness or scientific
qualification. Immutable predecessor lineage remains separately mandatory. No solver
algorithm or original numerical/acceptance limits changed.


## Captured host publication workspace and cold import

Nonlinear publication now carries the captured host allowance through original-root
authentication, each block projection and immutable-source persistence. Earlier public
history remains charged during authentication and recovery; the completed public payload
is retained in preparation admission through staging and commit. The source cannot enlarge
the allowance beyond its original authored ceiling. Codec, artifact and FEA resource causes
remain typed Capacity failures instead of invalid numerical-result errors.

Cold closure consumption rederives and validates the unchanged original execution plans,
responses and accounting, but does not execute those programs. Its private import workspace
is derived from the original unique object inventory using the shared canonical codec
projection, simultaneous transport/payload bounds and retained metadata. Execution still
reserves original task memory. The terminal adapter reuses the coordinating import reservation;
it rejects a missing reservation rather than double-charging the same cold workspace.

Both-order source tests reject zero/tiny host and exhausted retained-history allowances
before additional source reads, then authenticate and recover the untouched originals.
Four actual isolated runtime publishers cover both orders with/without final-field observations.
Insufficient host preparation exposes no children; unchanged default128MiB admission commits.
After publisher exit, cold filesystem reads preserve all original accepted outputs and verify
the independent uniaxial alpha law at every load factor. The original controller may retain11
accepted outputs, including its final endpoint; publication preserves that exact original count.
All three states remain present and publication remains blocked by mandatory spatial evidence.
Final native suite passes12/12 (107.62s), bridge186/186, runtime storage73/73; strict libraries,
native/bridge test lint, scoped format and architecture pass.

This is requested workspace admission and fixture lifecycle evidence, not measured RSS,
aggregate task/backend accounting, authored CAD qualification, nonuniform whole-runtime
publication or resolution of the seven original moving-front scientific failures. Broad runtime
library/test strict lint now passes after the2026-10-04 15:43UTC fixture reconciliation. Whole J2/atomic publication remains
uncommitted and unqualified; no mesh/reference/tolerance/resource ceiling was raised.


## Nonuniform whole-runtime admission discriminator

The independent aligned two-material Tet4/Tet10 witness now uses the original whole-study
coordinator runner, including field recovery, observation partitions/reduction/assembly,
accuracy, three semantic states and immutable publication. No alternate scheduler or native
kernel is introduced. The runtime publisher still executes in a separate process; cold
verification is authored for the exact original accepted load-factor sequence, unique output
field identities, displacement, six plastic components, alpha, von Mises and terminal stress.
Those new layered public-field assertions have **not** earned passing evidence yet.

The first layered case reaches all thirteen native phases and original-root validation but
fails actual runtime input authentication at the unchanged default128MiB host allowance.
The original import reservation summed temporary decode work for every sequential phase.
It now reuses the maximum whole-phase workspace, while coordinator state/fields/metadata/
transport and external caller storage remain independently charged. Cross-phase conflicting
reference identities and overflow still reject. Execution task plans and reservations are
unchanged; the shared codec worst-density projection is unchanged.

Full bridge188/188 passes, including the new peak/conflict/overflow tests. Refreshed native
suite is12PASS/1FAIL (195.76s); both homogeneous runtime witnesses and the independent layered
native trajectory pass. The new layered whole-runtime test remains a typed Capacity failure
before cold public field reads. Its logical authored-scope accounting falls from required
8840493000 to8628194218 bytes against8589934592; these counts include the offset holding the
unavailable authored allowance and are not actual8GiB allocation or RSS measurements.
The peak correction is necessary but insufficient for the128MiB publication scope.

Provider-enforced bounded verified reads are now available through shared `CacheImport`.
The default rejects unsupported providers; filesystem length is checked against the caller and
configured limits before allocation, followed by exact bounded reading, growth detection and
hash verification. Original native sources restrict reads to captured catalog lengths. Actual
shared canonical shape profiling admits original transport before reading or materialization;
manifest/partition inventory remains authoritative, without decoding or reconstructing physics.
Root raw decode buffers release after validation; transferred wire closures, pending metadata
and vector slot capacity remain charged. Shared suites pass189/189 bridge and29/29 artifact.

Structured admission diagnostics expose decoded, encoded, transport, caller and import
components while retaining the original typed capacity cause. They isolated a second overlap:
runtime captured preparation and source candidate admission charged the same original
completion. A builder-scoped retained-input token now transfers that existing charge through
authentication, materialization and persistence. Preparation stays charged for staging/commit;
foreign-builder and oversized inputs reject. Token regression passes and affected publication
checks pass42/42. Strict artifact/bridge/runtime/native library/test lint passes.

The layered witness now clears block replay but still fails at state replay: required8593858738
against8589934592 logical scope bytes (165.47s). No layered public-field pass is claimed.
The next draft derives temporary workspace for the current complete phase before preparation
and import, retaining global inventory validation and recovered-field workspace through
observation fanout. Its refreshed native witness fails1/1 (170.75s), still at state admission:
required8591994994 against8589934592, about2MiB over; no layered cold-field pass. Retained typed state/transport
remain separately charged; no original task reservation, codec factor or resource ceiling changes.

No mesh, reference, tolerance, publication decision or seven original scientific gates change.
Whole publication/J2 remain uncommitted and unqualified. Estimates are not measured allocator,
aggregate worker or RSS evidence; full lifecycle/candidate/scientific release gates remain open.

The next shared terminal lifetime draft separates completed accuracy host lineage from its
input/program buffers. Publication still consumes that exact host; buffers release after successful
import/accounting, with host metadata charged. Completed publication buffers also release only
after terminal evidence succeeds. Original accuracy/state/receipt/usage objects remain. Full native,
shared and strict verification is running; this is not a passing layered publication claim.

Completed-task release clears layered input authentication and all state replay. Full native
verification remains12PASS/1FAIL (248.36s), stopping at field materialization. Completion
retention now excludes future root-encoding work while preserving all original typed state,
response metadata and transport; encoding separately admits its full peak. This additionally
profiles original receipt/publication/replay/state-response metadata. Refreshed full native is
12PASS/1FAIL (119.35s), reaching FEA recovery preparation rather than prefix import. No layered
public-field assertions have passed. Shared189/189 and29/29, strict lint and architecture pass.

FEA recovery's retained-context capacity and overflow now remain typed resources, verified
by three preflight tests. The next demand must cover actual read-only physical validation,
assembly/projection and full outputs; the current entry still reserves the full nonlinear
pipeline envelope. That change and the refreshed typed witness remain pending. Original
caps, references and scientific gates remain unchanged; this is not measured resource evidence.


## Read-only original recovery admission

The current draft gives original accepted-history validation and field projection a
separate `StructuralNonlinearRecovery` phase and resource identity (estimation revision69).
It derives the same canonical integration, topology, properties, constraints, initial
physical assembly, recovery fields and observation inventory as execution. It reserves
both physical evaluations, pointwise history/equilibrium/scope validation, sparse support,
complete original carrier copies, field/history buffers and integration-plan construction.
The phase uses `NoSolve`: it advances no load target, runs no Krylov solver and constructs
no new solver artifact or state-profile catalog. Direct original recovery and portable
field-recovery workspace planning consume the same derivation.

Basis-authority encoding composes structural profiles of the original basis envelope,
actual compiled property identity, exact DOF row/node schemas, element tuples and
`PreparedPoint` identities/gradients. Inventory-only scalar values never enter physics.
The shared codec's multipliers are unchanged. Both Tet4/Tet10 and all five integration
rules pass actual complete-authority bound and unchanged encoded-identity checks.
In-memory preparation imports no serialized resource plan; the enclosing portable field
task independently reserves its own resource-plan transport and decoding.

Consuming cold projection releases already-validated terminal buffers before prefix replay,
moves original workload/mesh into the same coordinator, and retains exact original block
responses, terminal observations and their complete resource inventory. The returned prefix
artifact/checkpoint/receipt and wire remain charged. Runtime releases superseded terminal
runs before the next projection; original field history and immutable root are preserved.

Four preflight tests, five storage tests,169 resource regressions,189 bridge tests,29 artifact
tests,42 publication tests, five-package strict lint and132-member architecture checks pass.
Latest native suite passes12/fails1 (160.41s): first layered order publishes under the
unchanged128MiB allowance and reaches cold reads, but its witness compared only a default
256-value page against a288-value field. The corrected witness iterates bounded256 pages
and checks all original values, source identity, total/count/offset and truncation. Fresh
both-order verification is running; no full layered publication pass is claimed yet.
Original meshes, references, tolerances and seven scientific gates remain open. These
bounds and lifecycle tests are not measured RSS, aggregate/backend or release qualification.


The corrected bounded-page matrix passes all Tet4 layered cold references after publisher
exit, then fails Tet10 input authentication by1522746 logical scope bytes (full matrix
12pass/1fail,341.81s). Cold retention now removes a separately charged whole-coordinator
encoded copy that is never constructed during replay. Actual original transport, typed
structural bounds and current decoder workspace remain; completed-root encoding retains
its own full peak guard. Refreshed189 bridge/29 artifact tests, strict five-package lint and
architecture pass. The latest both-order native matrix is running; Tet10 and full lifecycle
qualification remain unproven. No resource ceiling, codec multiplier or numerical target changes.


The refreshed cold-only wire correction exposed a decoder/preparation overlap: a coarse
52,423,534-byte decoder estimate fit before task preparation, so profiling was skipped,
but prepared inputs subsequently exceeded the combined ceiling. Cold replay now always
profiles the original provider-bounded artifacts before preparing the task. Global inventory
validation remains; all actual transport, typed state, pending metadata and decoder workspace
remain charged. Refreshed189 bridge/29 artifact tests, five-package strict lint and architecture pass;
fresh native verification is running. No both-order pass is inferred
from the preceding12pass/1fail matrix or the completed Tet4 reference checks.


### Shared retained original semantic history

The three nonlinear semantic declarations retain complete original workload/block authority.
The draft shares identical payloads through immutable `Arc` after successful original state
validation and full-content comparison. Each declaration, identity and source association is
preserved. Serde wire encoding stays transparent; independently decoded equal payloads remain
separate until actual sharing occurs. Mutating an owned alias uses copy-on-write and therefore
produces a separately charged payload. Pointer sharing is a storage fact, never proof of physics.

The shared FEA retained-state inventory charges each actual payload allocation once and retains
all state metadata and control-handle allowance. Original wire objects, task inputs, receipts and
temporary decoding remain separately charged; producer wire-output allowance is unchanged.
Both-order cold state tests pass sharing, canonical-byte preservation and changed-history
rejection. The full original layered runtime witness remains live after the preceding Tet10
resource rejection (12 passed/1 failed,358.89s). Original limits and scientific targets remain
unchanged; no whole lifecycle, measured-memory or whole-study qualification is claimed.


The full shared-payload native matrix finishes12pass/1fail (362.99s): Tet10's original
state-retention admission overrun shrinks from7,437,005 to84,241 logical scope bytes.
The shared terminal tail now releases nonlinear accuracy context buffers after validated
accuracy/accounting import: later state tasks require workload and `source.result`, not the
completed accuracy mesh or duplicate result-input copy. Exact original workload/result and
all immutable phase/source associations remain. Latest exact both-order layered verification
is live; no full lifecycle pass is claimed. FEA semantic-state25/25, bridge189/189,
artifact29/29, strict five-package lint and architecture pass before this final lifetime edit;
refreshed five-package strict lint passes (4m03s), bridge189/189 and artifact29/29 pass,
and architecture passes after the edit. The original both-order native witness remains live.


The original both-order witness finishesFAIL1/1 (361.83s): initial authentication clears,
then Tet10 field materialization's fresh root import exceeds logical state retention by499,796
bytes. Runtime completion now exposes a complete retained inventory separating original metadata
from pointer-shared semantic payloads, using the same FEA state profiler. Native source admission
charges that inventory plus its separately owned root/catalog/access. Wire and decode budgets
are unchanged. New cold-source checks cover retained-bound-minus-one rejection and independently
redecoded equal states retaining separate allocation charges. Latest full native and strict lint
are running; architecture and whitespace pass. No full lifecycle/scientific promotion is claimed.


## Nonuniform element scratch and cancellation

Tet4/Tet10 physical assembly reuses fixed arrays for 30 columns and 30 products instead
of allocating B and C*B vectors at every integration point. Point and arithmetic order,
quadrature, sparse support, material laws and original point history are unchanged.
Scratch has 3,120 payload bytes on the current 64-bit native stack. This is not an RSS
measurement or whole-task bound. Existing resource formulas and limits remain unchanged.

Cancellation rejects before trial allocation, at every original integration point,
and around sparse tangent finalization. Newton preserves the typed cancellation error.
The allocation tracker observes zero retained and peak heap bytes during cancelled
evaluation, with committed history unchanged. Evaluation succeeds after context restoration.

A three-stage nonproportional path verifies both element orders. Closed polynomial
Tet10 derivatives and exact Tet4 corner interpolants provide all six strain components.
Separate point histories evolve from zero through the independently qualified constitutive
kernel using those closed strains. Stress, all six plastic-strain components and alpha
are compared at every original point. Assembly strain and interpolated history do not
supply that reference. Reference stresses drive direct tensor virtual work. Tangent action
is symmetric and matches central differences at h=1e-6 and h=1e-7, with relative error
below 1e-7 and the same committed history fixed. Tet10 must exhibit nonuniform point alpha.

These are kinematic and assembly checks using the existing point kernel. They do not
provide a new independent constitutive reference, continuum convergence proof or solution
to the original moving-front failures. Contact remains unimplemented and rejected; empty
contact outputs and a context contract do not supply contact search, enforcement or friction.
