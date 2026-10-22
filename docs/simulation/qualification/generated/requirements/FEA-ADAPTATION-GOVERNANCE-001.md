# FEA-ADAPTATION-GOVERNANCE-001: Accuracy-driven adaptation, persistent constraints and durable lineage

0 PASS, 3 PARTIAL, 0 FAIL, 47 OPEN, 0 TAGGED. Generated; do not edit.

## family_transfer_constraints

Evidence kind: AccuracyAcceptance. Criterion: Compose adaptive loops for each formulation with a validated estimator. Validate field and state transfer. Rebuild constraint plans from persistent identities and validate constraint-plan lineage after every h- or p-change.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| adaptive_analysis | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::adaptation::iteration::tests::constraint_lineage::persistent_clamp_rebuilds_new_midside_constraints_without_numeric_node_aliases` (crates/runmat-analysis/fea/src/adaptation/iteration/tests/constraint_lineage.rs:24) - Real Tet4 to Tet10 persistent face clamp rebuilds9 to18 rows, retains entity identity and rejects foreign/missing lineage. No reaction/state transfer or adaptiv... |
| adaptive_analysis | tet10 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::adaptation::iteration::tests::constraint_lineage::persistent_clamp_rebuilds_new_midside_constraints_without_numeric_node_aliases` (crates/runmat-analysis/fea/src/adaptation/iteration/tests/constraint_lineage.rs:24) - Real Tet4 to Tet10 persistent face clamp rebuilds9 to18 rows, retains entity identity and rejects foreign/missing lineage. No reaction/state transfer or adaptiv... |
| adaptive_analysis | tet4 | public_study | OPEN |  |
| adaptive_analysis | tet10 | public_study | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | native_process | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | durable | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | remote_execution | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | parallel_cpu | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | domain_decomposed | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## durable_accuracy_lineage

Evidence kind: AccuracyAcceptance. Criterion: Integrate adaptation graphs with durable and remote execution. Publish iteration lineage and target-quantity convergence in results. Drive stopping and acceptance through the shared accuracy plan, preserving family estimator evidence rather than inventing a separate adaptive tolerance model.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| adaptive_analysis | tet4 | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::adaptation::iteration::tests::governed::iteration_binds_canonical_solver_mesh_lineage_transfer_and_evidence` (crates/runmat-analysis/fea/src/adaptation/iteration/tests/governed.rs:95) - Canonical adaptation records use constructed converged solver facts and quantities, real mesh refinement/transfer fixtures; rejects missing/indeterminate eviden... |
| adaptive_analysis | tet10 | numerical_kernel | OPEN |  |
| adaptive_analysis | tet4 | public_study | OPEN |  |
| adaptive_analysis | tet10 | public_study | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | native_process | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | durable | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | remote_execution | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | parallel_cpu | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | domain_decomposed | OPEN |  |
| adaptive_analysis | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## release_convergence_transfer

Evidence kind: AccuracyAcceptance. Criterion: Add release evidence and documentation. Adaptation reduces the governed estimator and converges the selected target quantity. Transfer error remains below family-specific limits.

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

## constraint_state_lifecycle

Evidence kind: AccuracyAcceptance. Criterion: Constraint targets, relations, reactions, and state remain valid across adaptation. Cancellation, retry, restart, and replay preserve valid iteration lineage.

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

## constraint_lineage

Evidence kind: SemanticRealization. Criterion: Constraints bind to persistent geometry and topology identities rather than incidental node numbers. After h- or p-adaptation, the analysis projection rebuilds the canonical constraint plan, checks target coverage and pairing, transfers necessary state, and records old/new plan lineage.

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

