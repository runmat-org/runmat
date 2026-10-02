# Canonical steady incompressible fluid qualification

## Current envelope and decision

Experimental CPU FP64, serial, steady, homogeneous incompressible Newtonian flow
on canonical affine Tet4 cell/face geometry. This is not Stage 9 production
qualification. The exact realization cannot inherit the retired reduced-channel
operator's identity or maturity.

Authored fresh singleton steady Fluid studies use imported CAD, canonical meshing,
exact Cell material ownership, governed native tasks, original state and recovered
fields, authenticated cold completion, observations and runtime publication.
The standalone reduced-channel solver, dispatch and surrogate field construction
are removed. The direct model-only operation rejects with an authored-study-required
error. Reduced helpers still used by unfinished CHT/FSI are not canonical qualification.

Unsupported authored remote/GPU execution, transient flow, numerical checkpoint
resume, heterogeneous coefficients, turbulence, symmetry/periodic boundaries,
spatial/generated profiles and normal wall motion/ALE remain rejected.

## Equations and retained physical authority

The equations are div(U)=0 and
div(ρ U⊗U)=−grad(p)+div(µ(grad(U)+grad(U)ᵀ))+f.
Original fields are cell velocity U (m/s), pressure p (Pa), and owner-oriented
volume flux q (m³/s) on every face, including internal faces.

Shared sparse assembly and restarted GMRES replace neither the equations nor their
acceptance criteria. Momentum and reference-relative pressure corrections solve
Aδ=b−Ax around the current state. Compact Rhie–Chow pressure correction retains
original unrelaxed conservative fluxes. Physical gradients and linear-upwind
correction account for skew faces; unknown pressure traces at velocity boundaries
are reconstructed from actual neighboring samples.

Safeguarded Anderson depth one uses the shared physics-independent helper. A single
pressure-defect coefficient combines ordinary SIMPLE images of U, p and q without
mixing units in a norm. Its explicit default magnitude cap is32. Every candidate
is re-evaluated against original physical equations and retained only for strict
improvement of the maximum physical acceptance residual. Degenerate or excessive
coefficients, speculative overflow and invalid candidates retain the validated
ordinary image; outlet backflow clears history. No flux clipping, unit-scale
residual floor or reference-solution initialization is used.

All three signed constant velocity components, no-slip walls, tangential moving
walls, physical pressure outlets and volume body-force density are retained.
Each connected component requires a physical outlet; a gauge is not an outlet.
Outlet velocity has homogeneous normal gradient, not stress-free traction.
Temporary trial backflow uses signed extrapolated owner traces; accepted/recovered
fields reject backflow.

Shared incompressible-Newtonian laws require positive density and viscosity.
Original volume-element material ownership is exact Cell-role authority. Distinct
material IDs with identical coefficients are supported; stale, missing, wrong-kind
or substituted source/property maps fail closed.

## Independent numerical evidence

References are continuum expressions, never residual-generated forcing or fitted
RunMat outputs. Test boundary samples are frozen at original triangle centroids;
they do not establish authored spatial-profile realization.

- Couette: U=(y,0,0), p=0, ρ=µ=1; wall shear ±1 Pa.
- Affine convection: U=(1+y,0.2,0), p=−0.2x, f=0, ρ=µ=1;
  div(U)=0 and ρ(U·grad)U=(0.2,0,0)=−grad(p).
- Square duct: −µ(Uyy+Uzz)=G, transverse no-slip, G=µ=1.
  Independent odd Fourier sums carry explicit truncation bounds. At N=255,
  velocity tail ≤2/(π³N²), flow tail ≤8/(3π⁴N³), gradient tail ≤4/(π²N).
  Finite-series shear is not exact.

For duct refinements n=2,3,4, relative velocity L2 errors are
0.112188/0.075925/0.055199; flow errors0.191197/0.086351/0.048888;
wall-shear errors0.325795/0.213903/0.151078. All strictly decrease.
Global momentum imbalance is below4.2e−8 relative.

Affine convection errors are below1e−6 for velocity and1e−4 for physical pressure
on every tested mesh. These are affine reproduction/iterative-accuracy gates,
not an observed spatial convergence order. Non-affine duct gates retain strict
spatial refinement. Witness linear tolerance1e−11 and physical tolerance1e−7
remain unchanged.

Final isolated candidate verification: canonical CFD20/20, shared fixed-point5/5,
fluid-selected FEA35/35 (including one acoustic two-fluid test), full execution
bridge180/180, native prepared-closure and metadata preflight1/1 each, bridge
immutable source/resource authority1/1, runtime fluid publication-control test1/1.
CLI OCCT compilation, remote-family rejection1/1 and strict FEA/bridge/native
library/test lint pass.
Final native fluid tests pass6/6. The pinned authored lifecycle exited0, including
both CAD cases, cold replay/publication and policy/state/field/foreign-receipt tamper
rejection. The full FEA regression remains running with a reported transient-rod
time-refinement failure; no broad suite or release-lane green is claimed.

The original imported 1×2×3 m moving-box witness uses U=(0.001,0,0),
ρ=µ=1, target mesh1.5 m, outer cap400, inner1e−11 and physical1e−8.
The final pinned moving case accepted at315outer/137982linear iterations:
worst-cell momentum9.408899610537379e−9, global2.494419289565982e−9,
aggregate3.027216014114704e−13, continuity2.168404344973425e−15,
flux9.3989783610688e−15. The stationary case accepted at1outer/0linear iterations
with all five residuals zero. Both cold original runtime publications and terminal
lifecycle checks passed. Exact witness SHA256:
783f8413baeb1c86015a4b6cf68700bfcd2af8ac71e23a22e31992b81c393b0b.

## Governed execution, original-field recovery and publication

Resource-estimation revision61 includes correction vectors/actions and acceleration
history, proposals and physical recovery. Count-only geometry, numerical bounds,
decoded metadata, original source preparation and coexisting property maps are
admitted before expansion. Shared native plans govern solve, copy-only recovery,
state publication and final observations; requested bounds are not measured RSS.

Portable completion retains original model/step, policies, geometry, mesh, producer,
material map and canonical U/p/q bytes. Cold import authenticates digest before
decode, admits original-source preparation and decoded heap, and checks original
equations without solving again or replacing fields. Original exact state, result,
receipts, catalogs, phase accounting and observation target coverage are validated.
Partition observations import intersecting original Cell/all-owner-face blocks.
Missing mandatory spatial evidence stays Indeterminate and blocks production.

Runtime authenticates the entire scientific closure before publication and imports
the original recovered artifact under codec/transport/construction budgets, checking
bit-exact signed U/p/q. The generic prepared-study handoff validates original
root/revision and backend admission before using existing family coordinators.

Shared completion currently persists visible step runs before the family whole-study
manifest. A later storage failure can leave a visible run without a returned manifest.
No failure-injection or transaction proof closes this common-owner atomicity gap.
Immutable partial evidence files are not a successful whole-study completion.

## Historical failed trials and remaining gates

Before acceleration, the same outer400/inner1e−11 case failed worst-cell momentum
3.277707410e−8 against1e−8; tightening the inner tolerance did not close it.
Physical and assembled momentum vectors agreed within7.91e−19 N, ruling out that
suspected assembly/recovery discrepancy at the controlling cell. Late iterates
showed slow fixed-point contraction, not a proven rounding plateau.
Correction solves alone were insufficient. These failed trials are historical
diagnosis, not current qualification or permission to relax original gates.

An earlier main-source full lifecycle passed, but predates resource61 and the final
overflow safeguard; it cannot substitute for final candidate verification.

Measured memory/attempt usage reconciliation, independently satisfied mandatory
spatial evidence, generated/time-dependent inputs, symmetry/periodicity, transient
flow, numerical restart, partitioned/distributed algebra, GPU, conservative CHT/FSI,
publication atomicity and the blocking release lane remain open. The older runtime
fluid and operation-contract selections failed their FSI success expectations on
the retired directional fallback; no green coupled release lane is claimed.
No Stage9 or production gate is promoted by this bounded cutover.
