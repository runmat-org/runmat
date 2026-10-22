# FEA-FLUID-BOUNDARY-002: Complete fluid boundary and excitation set

0 PASS, 0 PARTIAL, 0 FAIL, 36 OPEN, 0 TAGGED. Generated; do not edit.

## profiles_and_excitations

Evidence kind: IndependentReference. Criterion: Velocity profiles (function, table, fully developed, periodic-spline sampled pulsatile waveforms from a CSV) and the `inlet_velocity` and `wall_motion` excitations run; square duct reproduces fRe 56.91 and circular Hagen-Poiseuille fRe 64.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | tet4 | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | tet4 | public_study | OPEN |  |
| incompressible_laminar_steady | tet10 | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | tet10 | public_study | OPEN |  |
| incompressible_laminar_transient | tet4 | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | tet4 | public_study | OPEN |  |
| incompressible_laminar_transient | tet10 | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | tet10 | public_study | OPEN |  |

## pressure_and_open_outlets

Evidence kind: SemanticRealization. Criterion: Pressure inlets and directional do-nothing open outlets with backflow stabilization run with flow reversing at the outlet; a closed pressure reference removes the null space without changing the solution.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | tet4 | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | tet4 | public_study | OPEN |  |
| incompressible_laminar_steady | tet10 | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | tet10 | public_study | OPEN |  |
| incompressible_laminar_transient | tet4 | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | tet4 | public_study | OPEN |  |
| incompressible_laminar_transient | tet10 | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | tet10 | public_study | OPEN |  |

## slip_symmetry_periodic_rotating

Evidence kind: IndependentReference. Criterion: Navier slip, symmetry, periodic boundaries with imposed pressure drop or flow rate, and rotating walls run: periodic channel flow rate is exact, Taylor-Couette torque matches u_theta = A r + B / r.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | tet4 | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | tet4 | public_study | OPEN |  |
| incompressible_laminar_steady | tet10 | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | tet10 | public_study | OPEN |  |
| incompressible_laminar_transient | tet4 | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | tet4 | public_study | OPEN |  |
| incompressible_laminar_transient | tet10 | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | tet10 | public_study | OPEN |  |

## fans_screens_porous_jumps

Evidence kind: PhysicalBalance. Criterion: Fan curves, screens and porous jumps apply a pressure jump consistent with their authored law and conserve mass.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | tet4 | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | tet4 | public_study | OPEN |  |
| incompressible_laminar_steady | tet10 | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | tet10 | public_study | OPEN |  |
| incompressible_laminar_transient | tet4 | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | tet4 | public_study | OPEN |  |
| incompressible_laminar_transient | tet10 | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | tet10 | public_study | OPEN |  |

## thermal_boundaries

Evidence kind: PhysicalBalance. Criterion: Inlet temperature, wall temperature, wall heat flux and convective wall conditions apply to the fluid energy equation and conserve energy.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| conjugate_heat_transfer | tet4 | numerical_kernel | OPEN |  |
| conjugate_heat_transfer | tet4 | public_study | OPEN |  |
| conjugate_heat_transfer | tet10 | numerical_kernel | OPEN |  |
| conjugate_heat_transfer | tet10 | public_study | OPEN |  |

