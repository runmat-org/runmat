# FEA-STRUCTURAL-SHELL-001: Physical membrane and bending shells

0 PASS, 1 PARTIAL, 0 FAIL, 27 OPEN, 0 TAGGED. Generated; do not edit.

## physical_shell_operator

Evidence kind: IndependentReference. Criterion: Select and document the supported shell formulation and assumptions. Implement membrane and bending behavior, local frames, thickness integration, drilling stabilization, mass, and recovery. Validate patch behavior, locking sensitivity, rigid-body modes, orientation, and mesh convergence.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | tet4 | numerical_kernel | OPEN |  |
| linear_structural | tet10 | numerical_kernel | OPEN |  |
| linear_structural | beam | numerical_kernel | OPEN |  |
| linear_structural | shell | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::assembly::elements::shell::qualification_tests::simply_supported_sinusoidal_plate_converges_to_the_navier_solution` (crates/runmat-analysis/fea/src/assembly/elements/shell/qualification_tests.rs:12) - Dense test assembly of transverse simply-supported triangular plate; center displacement decreases toward independent Navier value and fine error below8%. No me...<br>`runmat-analysis-fea[lib]::assembly::elements::shell::tests::shell_resultant_thickness_scaling_separates_membrane_bending_and_shear` (crates/runmat-analysis/fea/src/assembly/elements/shell/tests.rs:238) - Shell resultant thickness scaling separates membrane, bending and shear.<br>`runmat-analysis-fea[lib]::assembly::tests::unconstrained_shell_retains_its_physical_rigid_body_nullspace` (crates/runmat-analysis/fea/src/assembly/mod.rs:4735) - An unconstrained shell retains its physical rigid-body null space. |
| linear_structural | tet4 | public_study | OPEN |  |
| linear_structural | tet10 | public_study | OPEN |  |
| linear_structural | beam | public_study | OPEN |  |
| linear_structural | shell | public_study | OPEN |  |
| linear_structural | canonical_mixed_topology | native_process | OPEN |  |
| linear_structural | canonical_mixed_topology | durable | OPEN |  |
| linear_structural | canonical_mixed_topology | remote_execution | OPEN |  |
| linear_structural | canonical_mixed_topology | parallel_cpu | OPEN |  |
| linear_structural | canonical_mixed_topology | domain_decomposed | OPEN |  |
| linear_structural | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## proxy_removal

Evidence kind: IndependentReference. Criterion: Remove the pair-spring proxy when the formulation lands.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | tet4 | numerical_kernel | OPEN |  |
| linear_structural | tet10 | numerical_kernel | OPEN |  |
| linear_structural | beam | numerical_kernel | OPEN |  |
| linear_structural | shell | numerical_kernel | OPEN |  |
| linear_structural | tet4 | public_study | OPEN |  |
| linear_structural | tet10 | public_study | OPEN |  |
| linear_structural | beam | public_study | OPEN |  |
| linear_structural | shell | public_study | OPEN |  |
| linear_structural | canonical_mixed_topology | native_process | OPEN |  |
| linear_structural | canonical_mixed_topology | durable | OPEN |  |
| linear_structural | canonical_mixed_topology | remote_execution | OPEN |  |
| linear_structural | canonical_mixed_topology | parallel_cpu | OPEN |  |
| linear_structural | canonical_mixed_topology | domain_decomposed | OPEN |  |
| linear_structural | canonical_mixed_topology | claimed_accelerator | OPEN |  |

