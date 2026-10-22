# FEA-ACOUSTIC-MODAL-001: Acoustic cavity modal analysis

0 PASS, 0 FAIL, 12 OPEN, 0 TAGGED. Generated; do not edit.

## rigid_box_modes

Evidence kind: IndependentReference. Criterion: Rigid box modes equal c/2 sqrt((l/Lx)^2 + (m/Ly)^2 + (n/Lz)^2); the first 10 modes agree within 0.5 percent on Tet10.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_modal | tet4 | numerical_kernel | OPEN |  |
| acoustic_modal | tet4 | public_study | OPEN |  |
| acoustic_modal | tet10 | numerical_kernel | OPEN |  |
| acoustic_modal | tet10 | public_study | OPEN |  |

## impedance_walls_and_damping_in_modes

Evidence kind: IndependentReference. Criterion: Modal studies with impedance walls (headliner 1200 Pa s/m, rigid glazing) return complex or damped modes consistent with the harmonic response peak positions.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_modal | tet4 | numerical_kernel | OPEN |  |
| acoustic_modal | tet4 | public_study | OPEN |  |
| acoustic_modal | tet10 | numerical_kernel | OPEN |  |
| acoustic_modal | tet10 | public_study | OPEN |  |

## modal_publication

Evidence kind: PublicRun. Criterion: Modal runs publish frequency, pressure mode shape fields (one view per mode), residuals and normalization, pass resource admission on a cabin-sized model (3 cubic metre air volume, 12 modes), and the cabin study's first mode is reported.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_modal | tet4 | numerical_kernel | OPEN |  |
| acoustic_modal | tet4 | public_study | OPEN |  |
| acoustic_modal | tet10 | numerical_kernel | OPEN |  |
| acoustic_modal | tet10 | public_study | OPEN |  |

