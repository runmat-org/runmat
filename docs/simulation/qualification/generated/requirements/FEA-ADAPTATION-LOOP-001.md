# FEA-ADAPTATION-LOOP-001: Family-appropriate adaptive loops and separately qualified transfer

0 PASS, 3 PARTIAL, 0 FAIL, 27 OPEN, 0 TAGGED. Generated; do not edit.

## family_estimator_canonical_order

Evidence kind: SpatialConvergence. Criterion: Estimators must be formulation-appropriate and validated; a structural stress estimator is not a universal estimator. Adaptation decisions use stable identities and canonical evidence. Tet4/Tet10 order changes preserve correct field and integration-point state.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| adaptive_analysis | tet4 | numerical_kernel | OPEN |  |
| adaptive_analysis | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::adaptation::structural_recovery::tests::tetrahedron10_integration_point_estimator_preserves_linear_recovery` (crates/runmat-analysis/fea/src/adaptation/structural_recovery/tests.rs:274) - The Tet10 integration-point structural recovery estimator preserves a linear recovery; stress-jump and marking behavior are other tests. |
| adaptive_analysis | tet4 | public_study | OPEN |  |
| adaptive_analysis | tet10 | public_study | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | native_process | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | durable | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | remote_execution | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | parallel_cpu | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | domain_decomposed | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## qualified_transfer_governed_loop

Evidence kind: SpatialConvergence. Criterion: Plastic, contact, coupled-interface, and temporal state transfer must be separately validated before adaptation is enabled for those families. Topology change alone never constitutes convergence. Iteration limits, resource budgets, cancellation, retry, and publication fencing apply to the full loop.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| adaptive_analysis | tet4 | numerical_kernel | PARTIAL | `runmat-meshing-tetrahedron[lib]::cdt::adaptive_transfer::tests::quadratic::quadratic_refinement_preserves_all_polynomial_modes_and_cold_lineage` (crates/runmat-meshing/tetrahedron/src/cdt/adaptive_transfer/tests/quadratic.rs:52) - Quadratic refinement transfer preserves all polynomial modes and cold lineage for Tet4 and Tet10; plastic, contact and coupled-interface state transfer are not ... |
| adaptive_analysis | tet10 | numerical_kernel | PARTIAL | `runmat-meshing-tetrahedron[lib]::cdt::adaptive_transfer::tests::quadratic::quadratic_refinement_preserves_all_polynomial_modes_and_cold_lineage` (crates/runmat-meshing/tetrahedron/src/cdt/adaptive_transfer/tests/quadratic.rs:52) - Quadratic refinement transfer preserves all polynomial modes and cold lineage for Tet4 and Tet10; plastic, contact and coupled-interface state transfer are not ... |
| adaptive_analysis | tet4 | public_study | OPEN |  |
| adaptive_analysis | tet10 | public_study | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | native_process | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | durable | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | remote_execution | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | parallel_cpu | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | domain_decomposed | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## loop

Evidence kind: ControllerChronology. Criterion: Compose solve, family recovery, estimation, canonical marking, h/p adaptation, new mesh/lineage validation, primary field/history transfer, measured transfer error, resolve and estimator/target convergence into a governed continue/converge/reject orchestrator.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| adaptive_analysis | tet4 | numerical_kernel | OPEN |  |
| adaptive_analysis | tet10 | numerical_kernel | OPEN |  |
| adaptive_analysis | tet4 | public_study | OPEN |  |
| adaptive_analysis | tet10 | public_study | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | native_process | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | durable | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | remote_execution | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | parallel_cpu | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | domain_decomposed | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | claimed_accelerator | OPEN |  |

