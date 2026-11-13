# FEA-ACOUSTIC-MODAL-001: Acoustic cavity modal analysis

0 PASS, 6 PARTIAL, 0 FAIL, 6 OPEN, 0 TAGGED. Generated; do not edit.

## rigid_box_modes

Evidence kind: IndependentReference. Criterion: Rigid box modes equal c/2 sqrt((l/Lx)^2 + (m/Ly)^2 + (n/Lz)^2); the first 10 modes agree within 0.5 percent on Tet10.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_modal | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::modal::tests::linear_cube_converges_at_second_order` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/modal/tests.rs:144) - On a linear mesh the first nonzero cavity mode converges with an error ratio above 3 from 2 to 4 cells (second order); only the first nonzero mode is compared. |
| acoustic_modal | tet4 | public_study | OPEN |  |
| acoustic_modal | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::modal::tests::rigid_cube_modes_match_rectangular_cavity_including_degenerate_triplets` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/modal/tests.rs:65) - Eight modes of a rigid quadratic-mesh cavity on 2, 3 and 4 cells match the rectangular-cavity reference including degenerate triplets, with a zero-frequency uni... |
| acoustic_modal | tet10 | public_study | PARTIAL | `runmat[test:typed_fea_geometry_cli]::acoustic::response::rigid_cube_cavity_modes_match_rectangular_room_frequencies` (crates/runmat-cli/tests/typed_fea_geometry_cli/acoustic/response.rs:82) - A rigid unit cube on a quadratic mesh run through the CLI returns a zero-frequency uniform mode and three axial modes within 2 percent of c/2L = 171.5 Hz; the f... |

## impedance_walls_and_damping_in_modes

Evidence kind: IndependentReference. Criterion: Modal studies with impedance walls (headliner 1200 Pa s/m, rigid glazing) return complex or damped modes consistent with the harmonic response peak positions.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_modal | tet4 | numerical_kernel | OPEN |  |
| acoustic_modal | tet4 | public_study | OPEN |  |
| acoustic_modal | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::acoustic::canonical::modal::tests::pressure_release_ends_remove_the_rigid_mode_and_match_open_open_duct` (crates/runmat-analysis/fea/src/physics/acoustic/canonical/modal/tests.rs:109) - Pressure-release ends remove the rigid mode and five modes match the open-open duct within 4e-3 with exactly zero modal pressure on released nodes; impedance wa... |
| acoustic_modal | tet10 | public_study | OPEN |  |

## modal_publication

Evidence kind: PublicRun. Criterion: Modal runs publish frequency, pressure mode shape fields (one view per mode), residuals and normalization, pass resource admission on a cabin-sized model (3 cubic metre air volume, 12 modes), and the cabin study's first mode is reported.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| acoustic_modal | tet4 | numerical_kernel | OPEN |  |
| acoustic_modal | tet4 | public_study | OPEN |  |
| acoustic_modal | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-execution[lib]::tests::accuracy_plan_binding_tests::acoustic_response_accuracy_context_binds_the_authored_accuracy_plan` (crates/runmat-analysis/execution/src/tests/accuracy_plan_binding.rs:125) - The acoustic-response accuracy source context accepts the modal cavity result only under the authored accuracy plan and rejects a re-identified plan or loosened... |
| acoustic_modal | tet10 | public_study | PARTIAL | `runmat[test:typed_fea_geometry_cli]::acoustic::response::rigid_cube_cavity_modes_match_rectangular_room_frequencies` (crates/runmat-cli/tests/typed_fea_geometry_cli/acoustic/response.rs:82) - The modal run publishes modal.frequency_hz and five acoustic.pressure mode fields and is numerically acceptable; residuals and normalization are not asserted; a... |

