# Canonical steady incompressible fluid evidence

## Envelope and physical authority

This is an experimental CPU FP64, serial, steady, homogeneous incompressible
Newtonian envelope, not Stage 9 production qualification. Canonical affine
tetrahedral cell/face geometry carries original persistent boundary ownership.
The portable workload currently admits Tet4 only. Fields are physical cell
velocity (m/s), pressure (Pa), and owner-oriented face volume flux (m³/s).

The equations are div(U)=0 and div(ρ U⊗U)=−grad(p)+div(µ(grad(U)+grad(U)ᵀ))+f.
The operator uses shared sparse contribution assembly and shared restarted GMRES,
not a private matrix or a channel/profile correlation. Relaxed momentum and compact
Rhie–Chow pressure correction retain the original unrelaxed conservative face flux.
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
integration, measured RSS, process-separated restart or production qualification.

Native task scheduling, shared state publication, governed observations/accuracy/usage,
full authored runtime cutover, spatial/generated inputs, numerical restart,
transient flow, distributed/GPU execution, conservative CHT/FSI and release
qualification remain open. The old reduced production channel fallback has not
been removed by this slice. Runtime fluid checks pass 4/5: the existing FSI success
test depends on the reduced directional-excitation fallback now truthfully rejected.
The operation-contract suite similarly passes 47/48, with the FSI success contract
failing on the same directional fallback rejection. These coupled cutover failures
remain explicit; they are not standalone-fluid evidence or a green release lane.
No production gate is promoted here.
