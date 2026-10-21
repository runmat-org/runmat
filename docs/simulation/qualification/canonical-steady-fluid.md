# Canonical steady incompressible fluid evidence

## Envelope and physical authority

This is a CPU FP64, serial, steady, homogeneous incompressible
Newtonian envelope; Stage 9 qualification remains open. Canonical affine
tetrahedral cell/face geometry carries original persistent boundary ownership.
The portable workload currently admits Tet4 only. Fields are physical cell
velocity (m/s), pressure (Pa), and owner-oriented face volume flux (m³/s).

The equations are div(U)=0 and div(ρ U⊗U)=−grad(p)+div(µ(grad(U)+grad(U)ᵀ))+f.
The operator uses shared sparse contribution assembly and shared restarted GMRES,
not a private matrix or a channel/profile correlation. Relaxed momentum and compact
Rhie–Chow pressure correction retain the original unrelaxed conservative face flux.
The current draft solves linear corrections Aδ=b−Ax around the current velocity
and reference-relative pressure, using the same shared GMRES and authored inner
target. Its requested resource envelope includes the extra action and vector
workspace. Physical acceptance still evaluates the original equations, not the
correction-system residual.

The current safeguarded fixed-point draft uses the shared Anderson depth-one
proposal helper. One pressure-defect least-squares coefficient (bounded explicitly
by 32 in the default policy) combines the ordinary SIMPLE images of U, p and q.
It does not combine quantities of different units in a norm or initialize from a
reference solution. Every proposal is re-evaluated against the original physical
equations and retained only when the maximum physical acceptance residual improves.
Degenerate/excessive coefficients, speculative arithmetic overflow and invalid
trial fields retain the validated ordinary image; outlet backflow clears history.
The policy is explicit in serialized numerical identity, and additional history,
proposal and physical-recovery storage/work is admitted before solving. The
resource-estimation revision is 61. Requested bounds are not measured RSS.

Independent shared-helper tests pass5/5. Final canonical CFD tests pass20/20;
the focused acceleration suite passes5/5, including both Couette orientations,
original-equation cold recovery, policy substitution rejection, exact resource-
ceiling failures, speculative overflow, worsening merit and backflow rejection.
The final pinned authored lifecycle6194 completed with exit0 before retention in
`ab0cbb295`. Under original outer400/inner1e-11/physical1e-8 policy, moving CAD accepted
at315iterations; stationary used1outer/0linear and all five physical residuals0.
Cold original state/field/receipt retention, accounting replay, runtime publication
and missing/foreign/policy/precision/tamper rejection passed. These are scoped
lifecycle checks, not spatial or release qualification. Revision61 describes that
retained resource slice; the current shared model has subsequently advanced.
Requested bounds are distinct from measured RSS.

Physical gradient reconstruction and linear-upwind correction account for skew
faces. Unknown pressure traces at velocity boundaries are reconstructed from
actual neighboring pressure samples, not prescribed zero normal gradients.

Velocity constants preserve all three signed components. The initial kernel
supports explicit global constants, stationary no-slip and tangential moving walls,
physical pressure outlets and original volume body-force density. Each connected
component requires a physical outlet; a gauge reference is not one. Outlet velocity
uses homogeneous normal gradient, not a stress-free traction condition. Temporary
outlet backflow is permitted only in solver trials using the signed extrapolated
owner trace; original accepted and recovered fields reject backflow. Fluxes are
never clipped. Normal wall motion requires ALE and is rejected.

Assigned materials use the shared closed incompressible-Newtonian law, shared mass
density and positive dynamic viscosity. Exact Cell-role property compilation binds
each original volume element to its material. Equal coefficients with distinct
material IDs are supported; heterogeneous coefficients, missing owners, stale
retained domains and substituted source/property maps reject. Solid is not a Cell
alias. Unsupported frames, tables, turbulence, symmetry/periodic conditions,
transient/coupled procedures and checkpoint resume remain outside this envelope.

## Independent numerical witnesses

References are continuum expressions, never residual-generated forcing or fitted
RunMat outputs. Boundary samples are frozen at original triangle centroids; these
tests do not establish authored spatial-profile realization.

* Couette: U=(y,0,0), p=0, ρ=µ=1. Analytic wall shear is ±1 Pa.
* Nonzero affine convection: U=(1+y,0.2,0), p=−0.2x, f=0, ρ=µ=1.
  Both div(U)=0 and ρ(U·grad)U=(0.2,0,0)=−grad(p).
* Unit square duct: −µ(Uyy+Uzz)=G with no-slip transverse walls and G=µ=1.
  An independent odd Fourier series provides velocity, flow and shear, with
  explicit truncation bounds. At N=255, velocity tail ≤2/(π³N²), flow tail
  ≤8/(3π⁴N³), and gradient tail ≤4/(π²N). Finite-series shear is not exact.

For refinements n=2,3,4, the current duct relative velocity L2 errors are
0.112188, 0.075925, 0.055199; flow errors 0.191197, 0.086351, 0.048888;
wall-shear errors 0.325795, 0.213903, 0.151078. All strictly decrease.
Global physical momentum imbalance is below 4.2e−8 relative.

The affine convective velocity errors are 6.90e−9, 3.17e−8, 2.98e−8 and
physical-pressure errors 3.90e−6, 1.53e−6, 1.74e−6. These are affine
reproduction/iterative-accuracy evidence, NOT an observed spatial convergence
order. Every mesh must satisfy velocity error <1e−6 and pressure error <1e−4,
considerably tighter than the original coarse endpoint bounds. The non-affine
duct retains strict spatial-refinement gates. Solver tolerances are unchanged:
linear 1e−11 and physical relative 1e−7 for these witnesses.

Independent affine-pressure/skew-face tests, proper rigid rotation covariance,
weak/strong constant-Re forcing, local/global continuity and momentum, physical
tractions, original flux constitutive closure and invalid-input/resource tests
complete the focused kernel suite. A constant independent advecting field tests
affine face transport exactly; it does not claim exact quadratic face quadrature.

## Retained and execution boundary

`FluidSteadyWorkload` retains original model, producer step, geometry and numerical
policies. `FluidSteadySolverResult` retains original U/p/q, traction/residual
evidence, source/property/mesh/producer identities and rederived local reservations.
Cold import authenticates the expected payload digest before decode, admits original
source preparation and decoded heap, then checks the original equations without
solving again or replacing any field. The eight portable tests include zero flow and
independent nonzero rigid translation U=(1,0,0), p=0, q=S_x on every original face.
The latter verifies velocity and face flux against the analytic solution before
byte-exact cold roundtrip and solve-free original-equation recovery. It is not
process-separated numerical restart or spatial-profile qualification.

Count-only geometry/numerical bounds, metadata decode profiles and both coexisting
compiled property maps are admitted before projection/property expansion. Full
preparation additionally admits original authority-profile work and resident
storage: target aliases, signal comparisons and exact volume coverage. A valid
body-force target mixed with a missing or wrong-kind target rejects. Connected
component outlet admission uses shared labels and a single face pass; the focused
kernel suite is 13 tests, including these authority/resource regressions. Requested
heap estimates are not measured RSS or scheduler-attempt usage reconciliation.

The shared native-field slice represents U/p at original cells and q at
every owner-oriented face, including internal faces. It adds a distinct source-bound
finite-volume topology identity, not an exterior-only or nodal surrogate. Native
recovery copies the retained original fields unchanged, without a second solve.
Authored steady compilation preserves observation intent and exact Cell material
ownership; native final field/extrema requests are the bounded supported subset.
The streaming wire profile and source-authenticated resource projection are
requested-heap admission facts, not a shared scheduler plan or usage receipt.
Observation selection binds canonical cell order and complete exact source targets;
the source-authenticated verifier independently re-resolves original authored
coverage. Topology copies and metadata identity encoding are admitted before
allocation against explicit observation limits. Regression evidence covers wrong
cell ordering, partial targets, omitted/swapped plans, reversed owner orientation,
self-consistent reordered cold layouts and tiny caps. Focused fluid tests pass
23/23 and shared field/observation/accuracy results tests pass 140/140. These are
local field/observation and authored-intent evidence, not completed native lifecycle
integration, measured RSS, process-separated restart or Stage 9 qualification.

Governed native solve/copy-only recovery and complete original state publication
now use shared host resource plans. Scheduled final observations import only
intersecting original Cell/all-owner-face storage blocks, then run native
partition evaluation and reduction. Complete Fluid state retains the original
workload, mesh, canonical U/p/q result bytes and exact topology; cold validation
checks original equations without solving again. It is not a numerical restart
contract. The native cold-store chain verifies signed velocity, exact state,
receipt-backed algebraic/conservation evidence and missing mandatory spatial
evidence remaining `Indeterminate`. Requested bounds are not measured usage.
Current shared verification covers 31 focused fluid tests, 102 observation tests,
167 bridge tests and four native CFD tests. Stage 9 qualification remains open.

## Authored lifecycle cutover and historical diagnostics

Authored fresh singleton steady Fluid studies now prepare actual imported CAD,
canonical Tet4 meshing, exact Cell material ownership and governed native tasks.
The shared placement-neutral steady-study terminal authority retains original
Fluid state; it does not manufacture another profile-based solve output. Cold
completion reconstructs the same admitted phase chain, receipts and accounting.
Runtime publication imports the original recovered-field artifact under its
authenticated codec/transport/construction budget and verifies bit-exact signed
Cell U/p and every owner-oriented face q. It neither solves nor recovers them again.
Typed numerical and geometry resource policies replace reduced-channel sampling
controls. The exact canonical FV realization cannot inherit the old reduced operator's
identity or evidence.

The standalone reduced channel solver/dispatch and proxy field construction are
removed. The direct model-only operation rejects with an authored-study-required
error; it cannot fabricate a persisted Fluid run. Remote, GPU, transient and
checkpoint-resume authored Fluid roots remain rejected. Reduced components still
used by unfinished CHT/FSI paths remain; they are not canonical qualification.
Kernel tests pass16/16 and full shared execution tests172/172. FEA/bridge strict
lint and integrated libraries/CLI compile. The final authored CAD/cold-runtime process witness subsequently passed as
recorded above. Mandatory spatial evidence remains Indeterminate.

The generic prepared-study handoff validates original root/revision metadata and
backend/accelerator admission before using the existing family coordinators.
Its authentic preflight test and strict OCCT native checks pass. Current canonical
kernel selection passes14/14, including an exhaustion diagnostic that reports all
five physical acceptance residuals; strict FEA lib/tests checks pass.
The imported 1×2×3 m box lifecycle witness retains the original authored and
physical requirements1e-8, U=(0.001,0,0), rho=mu=1, and target1.5 m.
Before retained safeguarded acceleration, an explicitly prepared
outer400/inner1e-11 policy rejected worst-cell force
balance5.86653e-7. Inner1e-13 improves it to3.32597e-8; inner1e-14 plateaus at
3.27729e-8, still above1e-8. These are failed trials, not a qualified envelope.
Final fixture returns to explicit inner1e-11; defaults, resource caps, geometry,
load amplitude and physical acceptance are unchanged. Complete stationary cold
publication and tamper checks passed in preceding runs; the combined nonzero
lifecycle gate was open at that checkpoint and subsequently passed. No field is replaced or regenerated to make it pass.
The controlling cell's physical and unrelaxed assembled momentum vectors agree
within7.91e-19 N, far below its failed8.25e-14 N residual. Physical and algebraic
normalization scales differ about4001×. Independent maximum original-field errors
are8.65e-14 m/s velocity,9.03e-13 Pa pressure and2.60e-14 m³/s flux. This diagnostic
supports bounded convergence under low-Re force scaling, not an operator/recovery
mismatch. All temporary logging was removed with exact original-source parity.

The correction-solve draft passes the independent small-defect linear regression
(including zero defect and invalid inventory), and the original 14-test kernel
matrix. Before safeguarded acceleration, the authored moving case failed at
outer400/inner1e-11:
176,484 actual linear iterations, aggregate momentum1.06001e-12,
continuity7.15573e-15, flux3.11024e-14, global momentum8.74179e-9,
worst-cell momentum3.27771e-8 against the unchanged1e-8 gate. It reaches essentially
the prior tighter-inner plateau, so correction solves alone are not a demonstrated
qualification fix. The harness now chooses its multithreaded runtime once rather
than nesting runtimes. This removes a harness panic, not a physics failure.

Measured usage, spatial/generated inputs, numerical restart,
transient flow, distributed/GPU execution, conservative CHT/FSI and release
qualification remain open. Earlier runtime fluid checks passed4/5: the existing FSI success
test depends on the reduced directional-excitation fallback now truthfully rejected.
The operation-contract suite similarly passes 47/48, with the FSI success contract
failing on the same directional fallback rejection. These coupled cutover failures
remain explicit; they are not standalone-fluid evidence or a green release lane.
No qualification gate is closed here.


## Original pressure-state authority (2026-10-05)

The retained authored lifecycle uses zero-pressure outlets; its historical
unaccelerated failure is not caused by atmospheric-reference cancellation.
A separate scaled duct freezes accepted U/q, then adds101325Pa to its original
pressure and outlet. With rho, mu and force all scaled by1e-5, the independent
Fourier velocity/Re are unchanged. Absolute-pressure round trips lose7.0154e-12Pa
and worsen worst-cell closure7.3232e-8→8.4818e-7 under unchanged fixture1e-7.

The new implementation evolves and retains original governing pressure fluctuation
relative to each source-bound connected-component outlet. Absolute physical pressure
remains its finite f64 projection for field output, never restart/recovery authority.
The sparse equation, gradient, face action, relaxation, acceleration and original
recovery consume fluctuation directly. No pressure field is reconstructed or re-solved.
Cold recovery checks exact projection, full inventory, original input identity,
physical acceptance and retained residual. Result schema2/numerical identityv2 reject
old or missing governing state; model72 charges the additional retained vector and
shared structural codec profiling describes its actual numerical shape.

Fresh0/101325Pa solves pass with byte-identical U/q/fluctuation and iteration counts.
Cold state preserves it, and missing state, substituted absolute projection, foreign
source and reconstructed rounded fluctuation reject. Final focused tests2/2PASS3.50s
exercise both ordinary and safeguarded accelerated iteration. Canonical22/22PASS67.00s,
portable result/profile/field/rejection15/15PASS.13s, shared resources170/170PASS10.71s
and native5/5PASS3.23s cover worker output/receipt, cold original state, field recovery,
observations and publication. Strict FEA/native lib/tests lint passes. Main architecture
passes; isolated candidate retains two existing runtime rendering-size debts.
These are scoped implementation checks, not stage or measured-resource
qualification. Mandatory spatial accuracy, transient/coupled/claimed execution
and release gates remain open.
