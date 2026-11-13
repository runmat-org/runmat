# Canonical backward-Euler fluid interval

## Owner and equations

The analysis canonical finite-volume owner advances one physical interval using
backward Euler. For every original cell the unrelaxed momentum equation includes

```
rho V (u_end - u_start) / (t_end - t_start)
  + sum(oriented advective flux - viscous traction - pressure traction)
  - V body_force = 0.
```

The same inertia enters the shared sparse momentum operator, pressure-correction
diagonal, safeguarded fixed-point candidate assessment, independent original
cell/face/global balances, and cold recovery. SIMPLE underrelaxation is iteration
work; its extra diagonal does not become an original physical force. Pressure
remains the original source-bound fluctuation throughout iteration. Absolute Pa
is produced once as a reporting projection, never converted back for restart.

## Admission and original state

`prepare_backward_euler` borrows the original predecessor U, pressure fluctuation
and oriented q. It admits additional interval storage/work before traversing,
encoding or copying those fields. It rejects nonfinite, nonincreasing and
noncanonical times, invalid field dimensions/values, incompatible gauge means,
nondivergence-free original flux, nonfinite inertia, exceeded ceilings and
cancellation. The interval identity binds both exact physical times, every
original predecessor field and the prepared geometry/source/numerical authority.

Predecessor flux uses original incidence for incompressibility admission. It is
not tested against the receiving interval's boundary values: those can change
between intervals. Endpoint solve and recovery enforce the receiving exact
boundary laws. Admission does not silently repair initial divergence, replace
flux with a velocity-derived surrogate, or project the velocity onto a steady
solution.

Cold recovery recomputes the complete original equations from the original
predecessor and endpoint fields. A steady result, another interval, changed
predecessor, modified field or substituted residual cannot pass this recovery.
Identity agreement is provenance; independently recomputed balances supply the
physical evidence.

## Independent evidence

Uniform acceleration has the analytic solution `grad p = -rho a` with zero
viscous strain and zero advective divergence. Original affine boundary velocities
at the endpoint and a distinct original starting velocity exercise a genuine
inertial solve. The tests use zero and atmospheric pressure references, cold
serialization and changed-time/predecessor/field rejection.

Exponential uniform translation has `u = exp(-t) e_x` and
`grad p = rho exp(-t) e_x`, with no body force. Fresh endpoint boundaries on
successively halved intervals test first-order pressure time convergence against
that continuum derivative. Relative pressure-gradient errors at timestep
0.1/0.05/0.025 s are 0.05170871890262968/0.025422226002389654/
0.012605555018981468 at the unchanged terminal time 0.2 s. These tests do not use
a solver-generated reference. The exact added cell-mass action is independently
checked on every cell and all three axes against the analytic cell integral.

## Completion boundary

This is a numerical interval foundation, not a complete transient study or an
advertised native transient task. The existing steady authored/native/publication
path still uses its original steady equations and mandatory accuracy gate.

Shared signals, initial-state authority, time schedule and discontinuity semantics,
admitted temporal task blocks, full cold checkpoint lineage, cumulative work,
original histories, observations, mandatory time/spatial accuracy and atomic
publication must be connected before transient qualification. Existing
constant-source admission remains explicit; no family-private time profile,
scheduler or kernel callback is introduced. Transient vortex, cavity, backward
step, wider spatial/time and all supported execution evidence remain open.

The older CHT/FSI reduced-profile consumers remain unfinished work. This interval
must support their replacement with canonical conservative operators and original
field transfer; it does not qualify or promote those proxies.
