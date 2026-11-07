# FEA-SHARED-IDENTITY-001: Layered identity and closed chunked numerical artifacts

0 PASS, 2 PARTIAL, 0 FAIL, 14 OPEN, 0 TAGGED. Generated; do not edit.

## distinct_identity_layers

Evidence kind: PublicationIntegrity. Criterion: Use distinct canonical identities for: authored problem; semantic analysis sequence and procedure step; upstream state and state transition; discretized problem and solver mesh; constraint plan; operator structure; operator numeric values; numerical solve plan; observation and accuracy plan; execution and partition plan; attempt and worker-group epoch; and published result and evidence. CPU and GP...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-core[lib]::study::sequence::tests::excitation_selection_changes_solver_identity_and_roundtrips_distinctly` (crates/runmat-analysis/core/src/study/sequence/tests.rs:28) - Excitation selection changes solver identity and round-trips distinctly.<br>`runmat-analysis-fea[lib]::results::observation::tests::output_and_accuracy_changes_do_not_invalidate_solver_layers` (crates/runmat-analysis/fea/src/results/observation/tests.rs:2696) - Output and accuracy changes do not invalidate solver identity layers. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## closed_chunked_numerical_objects

Evidence kind: PublicationIntegrity. Criterion: Reuse the meshing system's architectural pattern of typed workloads, closed manifests, immutable content-addressed objects, chunking, verified transfer, retry proofs, and fenced publication. Do not duplicate its domain-specific types. Meshes, operator patterns and values, partition maps, fields, semantic states, histories, observations, accuracy evidence, preconditioner checkpoints, and solver che...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::execution::tests::numerical_objects_are_deterministically_chunked_and_self_validating` (crates/runmat-analysis/fea/src/execution/tests.rs:209) - Numerical objects are deterministically chunked and self-validating. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

