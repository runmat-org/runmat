# FEA-ACOUSTIC-TRANSIENT-001: Acoustic transient analysis

0 PASS, 0 FAIL, 8 OPEN, 0 TAGGED. Generated; do not edit.

## duct_pulse

Evidence kind: IndependentReference. Criterion: A one-dimensional pulse in a rigid duct matches d'Alembert travel time and amplitude to the declared tolerance.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_transient | tet4 | numerical_kernel | OPEN |  |
| acoustic_transient | tet4 | public_study | OPEN |  |
| acoustic_transient | tet10 | numerical_kernel | OPEN |  |
| acoustic_transient | tet10 | public_study | OPEN |  |

## energy_balance

Evidence kind: PhysicalBalance. Criterion: The transient energy balance (stored plus radiated plus dissipated minus supplied) closes to the declared tolerance, and time-step convergence is demonstrated.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_transient | tet4 | numerical_kernel | OPEN |  |
| acoustic_transient | tet4 | public_study | OPEN |  |
| acoustic_transient | tet10 | numerical_kernel | OPEN |  |
| acoustic_transient | tet10 | public_study | OPEN |  |

