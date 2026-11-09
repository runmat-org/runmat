# FEA-COUPLING-THERMOELASTIC-002: Thermal stress over a time history (thermomechanical transient)

0 PASS, 0 PARTIAL, 0 FAIL, 18 OPEN, 0 TAGGED. Generated; do not edit.

## time_history_transition

Evidence kind: IndependentReference. Criterion: Each structural step consumes the solved temperature of the matching thermal time (quasi-static, or with inertia in a structural transient); a uniform temperature ramp in a free bar reproduces alpha * dT expansion at every stored time, and a clamped bar reproduces -E * alpha * dT stress.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermoelastic | tet4 | numerical_kernel | OPEN |  |
| thermoelastic | tet4 | public_study | OPEN |  |
| thermoelastic | tet10 | numerical_kernel | OPEN |  |
| thermoelastic | tet10 | public_study | OPEN |  |
| thermal_transient | tet4 | numerical_kernel | OPEN |  |
| thermal_transient | tet4 | public_study | OPEN |  |
| thermal_transient | tet10 | numerical_kernel | OPEN |  |
| thermal_transient | tet10 | public_study | OPEN |  |

## transient_thermal_stress_reference

Evidence kind: IndependentReference. Criterion: Thermal stress under a surface heat pulse on a slab agrees with the classical one-dimensional transient thermoelastic solution (peak surface stress and its time) within the declared tolerance under refinement in space and time.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermoelastic | tet4 | numerical_kernel | OPEN |  |
| thermoelastic | tet4 | public_study | OPEN |  |
| thermoelastic | tet10 | numerical_kernel | OPEN |  |
| thermoelastic | tet10 | public_study | OPEN |  |
| thermal_transient | tet4 | numerical_kernel | OPEN |  |
| thermal_transient | tet4 | public_study | OPEN |  |
| thermal_transient | tet10 | numerical_kernel | OPEN |  |
| thermal_transient | tet10 | public_study | OPEN |  |

## time_history_publication

Evidence kind: PublicRun. Criterion: `thermomechanical transient` checks and runs from `runmat run`, publishes temperature, displacement and stress fields at every stored time, supports peak-over-time outputs and assessments, and snapshots select a time.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| thermoelastic | tet4 | public_study | OPEN |  |
| thermoelastic | tet10 | public_study | OPEN |  |

