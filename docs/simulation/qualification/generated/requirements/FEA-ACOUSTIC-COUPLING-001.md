# FEA-ACOUSTIC-COUPLING-001: One-way structure-borne sound

0 PASS, 0 PARTIAL, 0 FAIL, 8 OPEN, 0 TAGGED. Generated; do not edit.

## structural_velocity_drives_air

Evidence kind: IndependentReference. Criterion: An acoustic step reads a retained structural velocity state on a wetted surface (`normal_velocity on face = analysis.s.state.velocity` with `retain velocity`); a baffled piston matches the Rayleigh integral on axis, and the speaker study (cone harmonic, then 200 Hz, 1 kHz and 4 kHz radiation) reports on-axis SPL and directivity.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_structure_coupled | tet4 | numerical_kernel | OPEN |  |
| acoustic_structure_coupled | tet4 | public_study | OPEN |  |
| acoustic_structure_coupled | tet10 | numerical_kernel | OPEN |  |
| acoustic_structure_coupled | tet10 | public_study | OPEN |  |

## surface_transfer_conservation

Evidence kind: PhysicalBalance. Criterion: Volume velocity is conserved through the transfer between non-matching structural and acoustic surface meshes.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_structure_coupled | tet4 | numerical_kernel | OPEN |  |
| acoustic_structure_coupled | tet4 | public_study | OPEN |  |
| acoustic_structure_coupled | tet10 | numerical_kernel | OPEN |  |
| acoustic_structure_coupled | tet10 | public_study | OPEN |  |

