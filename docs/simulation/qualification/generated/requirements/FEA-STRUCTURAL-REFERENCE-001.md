# FEA-STRUCTURAL-REFERENCE-001: Linear structural independent references and physical balances

0 PASS, 3 PARTIAL, 0 FAIL, 25 OPEN, 0 TAGGED. Generated; do not edit.

## patch_reference_convergence

Evidence kind: IndependentReference. Criterion: Constant-strain and rigid-body patch tests. Cantilever, thick-cylinder, torsion, and multi-material interface cases. Tet4/Tet10 h- and p-convergence with expected rates.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::post::solid::tests::connected_tet4_affine_patch_recovers_analytic_strain_stress_and_energy` (crates/runmat-analysis/fea/src/post/solid.rs:424) - A connected Tet4 affine patch recovers analytic strain, stress and energy. |
| linear_structural | tet10 | numerical_kernel | OPEN |  |
| linear_structural | beam | numerical_kernel | OPEN |  |
| linear_structural | shell | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::assembly::elements::shell::laminate::tests::one_layer_membrane_patch_matches_the_independent_plane_stress_reference` (crates/runmat-analysis/fea/src/assembly/elements/shell/laminate.rs:434) - A one-layer membrane patch matches the independent plane-stress reference; no convergence series. |
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

## balance_independent_reference

Evidence kind: IndependentReference. Criterion: Load resultant, reaction balance, strain-energy, and reciprocal-work checks. Independent reference comparisons.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| linear_structural | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::assembly::solver_solid::tests::connected_tet4_affine_patch_balances_independent_boundary_tractions` (crates/runmat-analysis/fea/src/assembly/solver_solid/tests.rs:63) - A connected affine patch balances independent boundary tractions. |
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

