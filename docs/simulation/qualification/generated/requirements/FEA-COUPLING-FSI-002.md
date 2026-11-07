# FEA-COUPLING-FSI-002: Fluid loads on structures and two-way FSI

0 PASS, 0 PARTIAL, 1 FAIL, 15 OPEN, 0 TAGGED. Generated; do not edit.

## one_way_fluid_loads

Evidence kind: PhysicalBalance. Criterion: Pressure and wall shear stress from a solved flow map conservatively into a structural study as loads (total force and moment preserved), including wind-pressure-style mapped loads.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| fluid_structure_interaction | tet4 | numerical_kernel | OPEN |  |
| fluid_structure_interaction | tet4 | public_study | OPEN |  |
| fluid_structure_interaction | tet10 | numerical_kernel | OPEN |  |
| fluid_structure_interaction | tet10 | public_study | OPEN |  |

## two_way_ale

Evidence kind: IndependentReference. Criterion: Two-way partitioned FSI with Aitken relaxation, Laplace ALE mesh motion with geometric conservation and Newmark structure reproduces the Turek-Hron FSI1, FSI2 and FSI3 tip displacements and forces within the published scatter.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| fluid_structure_interaction | tet4 | numerical_kernel | OPEN |  |
| fluid_structure_interaction | tet4 | public_study | OPEN |  |
| fluid_structure_interaction | tet10 | numerical_kernel | OPEN |  |
| fluid_structure_interaction | tet10 | public_study | OPEN |  |

## compliant_channel

Evidence kind: IndependentReference. Criterion: A compliant channel reproduces lubrication-theory deflection within 5 percent and conserves interface work.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| fluid_structure_interaction | tet4 | numerical_kernel | OPEN |  |
| fluid_structure_interaction | tet4 | public_study | FAIL | `runmat[test:check_cli]::fsi_compliant_tube_example_runs_as_a_coupled_study` (crates/runmat-cli/tests/check_cli.rs:1237) - Steady compliant tube only; no transient ALE run. |
| fluid_structure_interaction | tet10 | numerical_kernel | OPEN |  |
| fluid_structure_interaction | tet10 | public_study | OPEN |  |

## large_deformation_solids_in_fsi

Evidence kind: PhysicalBalance. Criterion: The structural side of FSI accepts large-deformation solids with the same coupling residuals and conservation.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| fluid_structure_interaction | tet4 | numerical_kernel | OPEN |  |
| fluid_structure_interaction | tet4 | public_study | OPEN |  |
| fluid_structure_interaction | tet10 | numerical_kernel | OPEN |  |
| fluid_structure_interaction | tet10 | public_study | OPEN |  |

