# Portable fresh canonical fluid transient intent

## Source and formulation authority

`FluidTransientWorkload` retains the complete authored model, selected procedure,
canonical policy, geometry policy and original closed initial-field bindings. It does
not rewrite a transient procedure into a steady request. Canonical backward Euler has
its own temporal realization, separate from steady pressure correction.
Generic nonstructural admission cannot let profile code claim either canonical fluid
operator. Authored transient admission remains rejected.

Steady and transient workloads now share one bounded source compiler for original
geometry and Cell-role property/material authority. No second property map, fake
steady workload, mesh copy, reduced domain or scalar profile is introduced. Numerical
control validation also has one canonical owner. Shared initial sample-domain and
volume-region node selection are used by thermal and fluid; duplicate initial scalar
signal identifiers fail instead of choosing the first declaration.

## Fresh physical initial data

The selected before-step fluid velocity and pressure conditions support original
uniform values, quantity-aware signal samples and closed real global nodal artifacts.
Shared field admission binds quantity, shape, complete original mesh order and content.
Conflicting node declarations, missing/foreign artifacts, bad frame/domain and
nonfinite sources reject. Unassigned velocity is explicitly rest; unassigned pressure
is a zero fluctuation algebraic guess in the original source coordinate.

Affine Tet4 nodal velocity is integrated exactly to cell means and oriented triangular
face flux. It is not nearest-neighbor sampling or a guessed restart flux. Initial
continuity is checked on that original flux, without divergence repair or projection.
Original before-step data does not claim an equilibrated initial momentum solution or
an instantaneous imposed-boundary transition. A declared original absolute pressure
seed is converted once to its receiving reference coordinate. Reported absolute-Pa
fields are never used to reconstruct original solver state.

## Results, budgets and cold recovery

The portable result binds original workload, exact transient formulation, compiled
property identity/audit, original initial state and every original interval record.
Whole wrapper and block/result profiles precede execution and include original field
arrays, evidence, identifiers and transport materialization. These are requested
bounds, not measured RSS, scheduler reservation or cumulative native execution proof.

Cold decode yields claims. Recovery recompiles original source properties, re-realizes
original authored initial data and reevaluates every original interval. Changed
initial state, source/material/formulation, ordinals or fields cannot be authenticated
by updating hashes. No solve, interpolation or numerical repair occurs during recovery.

## Independent evidence and unresolved accuracy

An authored shared velocity ramp and initial signal sample preserve original pressure
and velocity through fresh execution, canonical transport and cold recovery. Uniform
acceleration and the gauge-anchored affine pressure reference are independently known.
A nonuniform original nodal rotation artifact reproduces exact cell means and face
integrals; a divergent artifact is rejected. The latter is initialization evidence,
not a qualified rotating transient continuum envelope.

The first acceleration witness using default linear tolerance 1e-8 failed the unchanged
physical continuity target 1e-8 at about 2.39e-8 after 200 iterations. That evidence
remains recorded. Retaining the existing authored fixture's linear tolerance 1e-10
passes the same reference, physical target and cap. This does not establish universal
default-control accuracy or replace mandatory spatial/time studies.

## Remaining completion boundary

The workload is fresh local numerical intent. Native transient task/resource phases,
immutable original checkpoints, cumulative cross-block work/output ordinals, process-
terminated continuation, observation/history artifacts and atomic publication remain.
Selected object activation, inlet/wall/mass-flow and generated input realization,
full typed cell/face restart authority, mandatory spatial/time accuracy and independent
vortex/cavity/backstep qualification are still required. Reduced CHT/FSI physics must
be replaced with canonical original-field operators and conservative transfer. No
qualification, coupled, distributed, accelerator, measured-resource or release promotion.
