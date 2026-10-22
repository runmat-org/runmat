# FEA-EM-TRANSIENT-001: Transient electromagnetics

0 PASS, 0 PARTIAL, 0 FAIL, 6 OPEN, 0 TAGGED. Generated; do not edit.

## step_current_diffusion

Evidence kind: IndependentReference. Criterion: A step current into a conducting slab diffuses with the analytic error-function series (field and current density) within the declared tolerance, converging in time step and mesh.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| electromagnetic_transient | edge_elements | numerical_kernel | OPEN |  |
| electromagnetic_transient | edge_elements | public_study | OPEN |  |

## switching_and_pulse_loads

Evidence kind: PhysicalBalance. Criterion: Pulsed and switched voltage or current drives from authored time signals run and conserve energy: supplied equals stored magnetic energy plus Joule loss plus work.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| electromagnetic_transient | edge_elements | numerical_kernel | OPEN |  |
| electromagnetic_transient | edge_elements | public_study | OPEN |  |

## loss_to_heat

Evidence kind: PhysicalBalance. Criterion: The Joule loss history is available to a thermal transient as a conservative heat source without remeshing.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| electromagnetic_transient | edge_elements | numerical_kernel | OPEN |  |
| electromagnetic_transient | edge_elements | public_study | OPEN |  |

