# Shared timed inputs for canonical fluid intervals

## Ownership and physical realization

`CanonicalFluidProblem::prepare_at` takes the shared `ProjectionSample`; the shared
excitation owner converts it to its typed sample point. The fluid owner accepts
static or physical time inputs, including the explicit one-sided approach. It does
not define a time-profile format, sampler, interpolation rule, scheduler or callback.

Full-vector prescribed velocity, moving-wall velocity, outlet pressure, gauge
pressure and body-force density use the existing quantity-aware evaluator. Authored
constant, tabulated and waveform representations follow that evaluator's domain,
interpolation, extrapolation and discontinuity semantics. An incompatible frame,
shape, quantity, nonreal component, nonfinite value or unresolved artifact/generator
is rejected. Existing geometry, coverage, flux, moving-wall/ALE and body-force target
restrictions remain. A receiving timed input must match the interval's physical
endpoint exactly. Timed preparation cannot call ordinary steady solve or recovery:
it requires the explicit original predecessor interval. One-sided boundary-law
inspection is not a steady result masquerading as a transient solve.

The shared discontinuity collector now includes active fluid velocity, moving-wall,
pressure-reference and outlet-boundary signal IDs. Constraint/boundary activation
uses the procedure's existing object-selection authority. The collector still owns
only event selection; it does not supply another schedule.

Shared prescribed scalar lookup rejects duplicate signal IDs instead of selecting
the first declaration. Prescribed helpers have one focused module under the shared
sampler. Vector and directional-vector sampling reject nonfinite components.

## Original pressure coordinates and cold data

Original results retain all three quantities independently:

- governing pressure fluctuation;
- source-bound component pressure reference;
- rounded absolute-Pa reporting projection.

Original references are retained directly from preparation, never inferred by
subtracting fluctuation from rounded absolute pressure. Original field views borrow
these references with U/p-fluctuation/q. Interval identity includes that complete
predecessor inventory. Pressure is an algebraic unknown at the receiving time;
its original fluctuation is the warm guess in the receiving pressure coordinate,
not a pressure history reconstructed by an absolute-Pa round trip.

Cold recovery rederives the source reference, compares the retained reference,
checks its exact absolute projection, and independently reevaluates the original
physical equations. Missing or substituted references and obsolete result schema
are rejected. Steady retained-result schema is 4; numerical-input domain is v6 and
backward-Euler interval domain is v2. These contracts do not accept a
legacy payload with missing original coordinates as restart authority.

## Resources and independent evidence

Admission streams the actual original signal payload through the shared canonical
codec profile before physical-signal validation, identity materialization or
sampling. Long unused tables are still original source authority and are charged.
The original reference array is included in numerical/output profiles. Resource
estimation revision is 77. Requested envelopes are not measured RSS or scheduler
allocation evidence.

A fresh uniform-acceleration case uses a shared velocity table and gauge table that
changes from 0 to 1e16 Pa. The independently known gradient remains `-rho*a` even
though every absolute-Pa output rounds to 1e16. The original nonzero fluctuation,
reference and gradient survive cold recovery; altered reference is rejected.

Other tests cover one-sided velocity on exact original faces, source identity
separation, volume-integrated time-varying body-force resultant, extrapolation and
domain rejection, duplicate IDs, large-table resource rejection, scalar jump sides
and active fluid discontinuity inventory. Original duct, rotational and temporal
refinement gates remain unchanged.

## Remaining completion boundary

This slice supplies sampled numerical inputs and complete pressure coordinates.
The governed authored transient workload, selected excitation cases and step
activation, typed initial conditions, shared procedure-time grid with discontinuity
arrivals, admitted temporal blocks, cumulative work/output ordinals, cold native
checkpoint chain, histories, observations, mandatory spatial/time accuracy and
atomic transient publication remain required.

Inlet/wall/mass-flow excitation actions and generated input materialization must
also be connected through their shared authorities; the current canonical kernel's
explicit body-force-action restriction remains visible. A one-sided input test does
not qualify time integration across a velocity jump or an impulsive boundary change.
No transient native task, CFD, CHT, FSI, accelerator or distributed
qualification is inferred from these numerical and steady-native checks.
