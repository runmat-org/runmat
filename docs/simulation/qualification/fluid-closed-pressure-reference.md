# Closed-component fluid pressure references

An authored `PressureReference` supplies a gauge, never an outlet or traction law.
Its value is the area-weighted reconstructed pressure trace on the exact selected
boundary faces, separately for each connected component reached by the target.
Every exterior face still requires its original physical velocity/wall law.
Closed components need exactly one gauge authority; physical-outlet components
reject additional gauges. Signals must resolve to authenticated study constants.

Shared algebra validates the original symmetric graph operator's constant mode
and compatible original source before numerical anchor-row reduction.
Source-construction roundoff is bounded from actual face-term magnitudes, separately
from the cancellation-small RHS; original equations remain unchanged and all
physical acceptance checks retain their existing tolerances. It does not
subtract a mean source, repair flux or discard an incompatible equation. The solver
uses existing GMRES. The reconstructed pressure is shifted by a component constant
into the authored mean-trace gauge; no pressure gradient or physical flux is changed.
Recovery verifies the gauge and every original conservation equation. Gauge IDs
are not fabricated physical-boundary force reports.

Numerical input v5 and resource revision76 bind gauge authority and account for
selection/component arrays, original/reduced systems and normalization workspace.
Cancellation is honored by outer iteration and shared Krylov execution.
These are requested bounds, not measured RSS. Closed-domain analytic translation,
constant pressure and corruption/authority/compatibility tests establish the tested
CPU envelope. General cavity accuracy and mandatory spatial evidence remain open;
this is not Stage 9 qualification.

A closed pressure reconstruction still needs an affinely spanning cell-centroid
inventory. The original six-cell cube has coplanar centroids and is rejected;
a gauge does not invent the missing pressure gradient. Closed reference tests use
spatially sufficient refined meshes. Existing open-flow mesh gates are unchanged.

## Independent nonconstant pressure evidence

Original affine boundary velocities for rigid rotation `u=(-omega*y,omega*x,0)`
with `omega=0.01`, density=viscosity=1 and zero body force drive fresh closed solves.
The independent centrifugal pressure gradient is `omega^2*(x,y,0)`; it is never
used as a solver RHS or manufactured result. On cells2/3, volume-weighted relative
velocity errors decrease `2.11459e-5 -> 5.50042e-6`, and pressure-gradient errors
`0.250844 -> 0.141097`. Original physical balances and mean-trace recovery pass.
This is a bounded library/reference envelope, not lid-driven cavity qualification
or a generally certified per-study spatial estimator.

The native cold chain also executes closed-wall/gauge workloads through original
solve/receipt, all-face field recovery, observations, complete state and accuracy
publication. Missing mandatory spatial evidence remains Indeterminate.
