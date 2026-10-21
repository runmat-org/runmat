# Canonical nodal body density — qualification incomplete

This Stage 3/4 slice realizes frozen physical vector-density samples before
assembly. It does not qualify the structural family or establish authored
mesh-dependent generators, field-artifact execution, restart or remote equivalence.

## Ownership and supported numerical envelope

Shared signal contracts explicitly declare topology, field location and component
count. The current structural consumer admits canonical-node vectors with three
components, BodyForceDensity units, and the exact sealed solver-mesh digest.
Every constant/table/waveform tensor must have `[mesh_nodes, 3]` shape and complete
rows in canonical mesh order. Complex samples are frozen before component selection.
Foreign meshes, incomplete values and unsupported association are rejected.
No callback runs inside an element quadrature loop or numerical solve.

Tet4/Tet10 volume projection interpolates the density with the element shape
functions and integrates `F_i = integral(N_i b dV)` with physical volume measure.
Formulation evidence is CanonicalVolumeProjection, not equal-node distribution.
Explicit beam/shell and synthetic projection do not inherit this admission.
Concrete coordinate frames are transformed at physical quadrature points.

The unchanged sampled-value identity contract limits encoding to 1 MiB,
collections to 1,024 items, strings to 1,024 bytes and nesting to 32. Consequently,
this unpartitioned field envelope rejects more than 1,024 node rows rather than
silently truncating fields or increasing a limit. Larger fields need an explicit
governed chunked representation and resource path.

## Independent physical evidence

For a cube `[0,L]^3` and density `b=(x,2y,3z)`, the independent volume integrals
give resultant `L^4*(1/2,1,3/2)` and moment about the origin
`L^5*(1/4,-1/2,1/4)`. Component-wise virtual work against `(x,y,z)` gives
`L^5*(1/3,2/3,1)`. Tests exercise nonuniform Tet4/Tet10 subdivisions at two
length scales, not an equal-node-force reference. The two quadrature/authority
tests pass on the isolated first resource revision-48 snapshot.

A separate prospective time-domain test uses the longitudinal elastic equation
and stress law documented by the independent
[UT Austin longitudinal-wave derivation](https://farside.ph.utexas.edu/teaching/315/Waves/node34.html).
For unit cross-section and length, density 1, E=100, nu=0, constrained transverse
motion and fixed axial displacement at x=0, the following Rayleigh-damped
manufactured extension is derived directly from continuum equilibrium:

```
q(t) = A cos(omega t), A=0.01, omega=2*pi
u(x,t) = (x*q(t),0,0)
b_x(x,t) = rho*x*(q''(t) + alpha*q'(t))
traction_x(1,t) = E*(q(t) + beta*q'(t))
U = E*q^2/2; T = rho*(q')^2/6
D = (alpha*rho/3 + beta*E)*A^2*omega^2
    *[t/2 - sin(2*omega*t)/(4*omega)]
W_external = U + T + D - U(0)
```

Undamped and `(alpha,beta)=(0.8,0.002)` cases use physical density/traction
signals, not forcing manufactured from RunMat matrices. Initial displacement
is exactly affine and initial velocity zero. Fixed Tet4/Tet10 meshes use
20/40/80 time intervals over 0.2 seconds. U/V/A and elastic/kinetic/dissipated/
external-work histories must improve by more than three per refinement and
finish below 0.0002 normalized RMS; support work is zero and discrete closure
is below 1e-6. These gates were specified before execution.

The corrected revision-48 snapshot passes this complete four-case test in
13.76 seconds. Finest normalized acceleration RMS errors are 3.20243e-5 and
3.06871e-5 for Tet4 undamped/damped, and 2.78013e-5 and 2.65006e-5 for Tet10.
Every nonzero measured quantity meets both refinement gates and the finest bound;
zero undamped dissipation, support work and discrete closure also pass.
The initial invocation rejected stale formulation-source compatibility before
integration; the corrected admission names actual physical volume projection.
This is spatially affine temporal evidence, not general spatial convergence.
Existing free-decay spatial and acceleration failures in solid-rod-transient.md
remain unchanged and open. The iterator-only Clippy correction also passes the
numerical rerun in 14.03 seconds, with unchanged errors; no qualification or
whole-study execution gate is promoted.

## Resource and remaining execution evidence

One sampling-free active-action authority counts each frozen field, including
multiple actions sharing a signal. It reserves the complex payload sum plus the
maximum transient sampling/clone/encoding workspace, once per task, not once per
time sample. Static, frequency, modal and transient planners share these typed
facts; transient block phase replacement preserves them. Encoding projection
uses the shared canonical profiler without allocating node arrays. First-snapshot
workspace, encoding and additive-residency/cold-plan tests pass. Final helper and
negative-test graph passes six workspace, two encoding, two residency/cold-plan,
nine admission and two quadrature/authority tests. Final strict selected-package
Clippy passes for core, FEA and analysis execution, including library tests.
Native library compilation also passes; neither check establishes native
generated-field execution. Estimates are not measured peak-process evidence.

Actual post-mesh callable admission, sealed coordinate/ordered-row inputs,
portable chunk execution, artifact/native/cold equivalence and independent
whole-study accuracy/restart/placement/release evidence remain required.
