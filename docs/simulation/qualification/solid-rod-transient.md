# Solid rod transient qualification — not qualified

This bounded Stage 4 qualification connects physical Tet4/straight Tet10 solid
assembly, direct second-order integration, authored initial conditions, block
continuation and recovered histories to an independent continuum solution.
It does not qualify the whole formulation.

## Independent problem

The unit cube has unit density, `E=100 Pa`, `nu=0`, and unit cross-sectional area.
Transverse translations are constrained; axial translation is constrained at
`x=0`, and the end at `x=1` is traction free. These conditions admit the exact
three-dimensional field `(u(x,t),0,0)` without a slenderness approximation.
The axial wave equation and fixed/free spectrum follow the same independent
[longitudinal-wave reference](https://farside.ph.utexas.edu/teaching/315/Waves/node34.html)
used by the harmonic rod qualification.

Initial displacement is `A sin(lambda x)`, initial velocity is zero, `A=0.01 m`
and `lambda=pi/2`. No external force acts. With Rayleigh coefficients `alpha`
and `beta`, the exact amplitude satisfies:

```
omega_n² = E lambda² / rho
gamma = (alpha + beta omega_n²)/2
omega_d = sqrt(omega_n² - gamma²)
q(t) = A exp(-gamma t) [cos(omega_d t) + gamma/omega_d sin(omega_d t)]
q_dot(t) = -A exp(-gamma t) omega_n²/omega_d sin(omega_d t)
q_ddot(t) = -2 gamma q_dot(t) - omega_n² q(t)
u(x,t) = sin(lambda x) q(t)
```

The two cases are undamped and `alpha=0.8 s^-1`, `beta=0.002 s`.
Integration extends to `0.2 s`, one undamped half-period. Spatially integrated
continuum elastic and kinetic energies are `E lambda² q²/4` and `rho q_dot²/4`.
Cumulative damping dissipation equals initial energy minus their sum. These
references use no RunMat stiffness, mass, damping matrices or eigensolve.

## Authored and verification boundaries

The spatial initial displacement is authored as an exact typed per-node initial
condition using persistent canonical node selectors. Separate retained original-field
witnesses exercise `InitialValue::FieldArtifact` transport and cold admission; those
checks do not replace this original continuum gate or qualify generated inputs.

Spatial refinement holds 400 time intervals fixed and uses Tet4 cells 2/4/8 or
straight Tet10 cells 1/2/4. Temporal refinement holds the four-cell Tet10 mesh
fixed and uses 10/20/40 intervals. No simultaneous refinement hides either error.
Every node and time sample contributes to displacement, velocity and acceleration
RMS errors normalized to fixed physical peaks. Energy errors use initial continuum
energy, never a vanishing instantaneous value. Zero applied/support work and
discrete mechanical closure are checked separately.

Preset spatial gates require error reduction by 1.6 per Tet4 refinement and 3 per
Tet10 refinement, with finest normalized RMS errors below 5% and 0.5%, respectively.
Temporal displacement/velocity errors must decrease by three per refinement and
finish below 0.2%. The independent average-acceleration phase estimate
`omega_n³ T dt²/12` is approximately `1.6e-5 rad` at 400 intervals and
`0.001615 rad` at 40 intervals. Solver residual alone cannot satisfy these gates.

The cold execution case uses two blocks on eight-cell Tet4/four-cell Tet10 meshes,
80 intervals, and the damped solution. It decodes workload, mesh, requests,
segments, checkpoint, assembled result and numerical receipt, then recovers the
physical histories. Histories must meet the continuum bounds as well as match
uninterrupted execution. Foreign initial conditions/topology cannot reuse the
result's recovery admission. This is cold portable-object continuation, not an
actual failed-process restart, remote worker replacement or distributed solve.

## Verification state

Compilation and the analytic initial-state/energy test pass. The first undamped
temporal velocity-convergence gate fails on the fixed four-cell Tet10 topology:
normalized velocity errors are `0.00864145`, `0.00269245`, and `0.00180031` at
10/20/40 intervals. Displacement errors decrease to `0.000435004`, but acceleration
errors remain near `0.03314`; this is not a qualified full-field envelope.
The cold-history test also fails the unchanged Tet10 acceleration gate:
normalized RMS `0.00749137 > 0.005` (U `0.000133870`, V `0.000390312`,
energy `0.0000994941`). Its Tet4 branch completes the independent bounds,
uninterrupted U/V/A comparison and authority negatives before that failure;
the whole test is not a pass. Spatial refinement also fails: undamped Tet4
acceleration errors `0.405261`, `0.206054`, and `0.100025` over cells2/4/8
decrease about first order, but the finest value exceeds the unchanged 5% gate.
U errors `0.047586/0.011082/0.002573`, V `0.099115/0.028455/0.007481`, and
energy `0.032318/0.008527/0.002177` decrease; they do not compensate for the
failed mandatory acceleration bound. Later damped/Tet10 spatial cases are not
reached. The invocation settles with one analytic test pass and three failures
after 1761.93 seconds; this envelope remains unqualified.

Interpolating the initial displacement does not make it an exact discrete
eigenvector: higher spatial modes can contribute disproportionately to velocity
and acceleration. That is a diagnostic explanation, not acceptance evidence or
permission to subtract a spatial floor from the independent reference. The
spatial results reveal first-order acceleration behavior, but do not yet establish
whether all observed errors follow the expected initial-field regularity envelope.
No qualified replacement initial-state envelope is claimed.
No tolerances, mesh refinements or proof limits have been relaxed after failures.
The separately documented [affine forced transient](structural-affine-transient.md)
isolates temporal integration, reaction/momentum balance, and cold histories using
a spatially exact manufactured field. Its passing checks do not close these
free-decay spatial or initial-field qualification failures.
The separate damped-modal Tet4 coercivity-proof blocker is neither retried nor
bypassed by this direct transient case.

```sh
cargo test --locked -p runmat-analysis-fea --lib pipeline::transient::rod_qualification -- --nocapture
```

General bending/shear, unconstrained Poisson response, curved geometry, generated
input equivalence, formal study-accuracy acceptance, measured resource envelopes,
process/remote restart and final release evidence remain separate required gates.

## Exact affine consistent mass

Affine Tet10 elements now integrate consistent mass directly from barycentric
polynomial moments. For exponents `a_i`, the physical tetrahedron integral is
`6 V product(a_i!) / (3 + sum(a_i))!`. Quadratic corner and edge shapes yield
integer coefficients times `rho V / 420`; the negative corner/edge couplings are
retained. This is consistent mass, with no row-sum lumping or mode fitting.

The existing exact midpoint predicate selects this path. Positive Jacobian and
material admission run first; curved elements, including a `2^-40` midside
perturbation, retain the original positive 125-point rule. The affine path evaluates
one constant determinant and the exact coefficients, eliminating unnecessary
quadrature accumulation. Existing requested resource ceilings remain conservative;
this is not a measured performance or RSS claim.

Independent tests cover every quadratic kinetic-energy cross moment across three
scales and three origins, including `2^40`, all vector axes, symmetry and cross-axis
zeros. Curved element mass is compared against an independent 512-point integral.
No original continuum reference, mesh, tolerance or initial displacement changes.
The original temporal gate was revalidated before this change and still fails its
velocity refinement ratio; exact mass is not claimed to resolve that scientific gap.
