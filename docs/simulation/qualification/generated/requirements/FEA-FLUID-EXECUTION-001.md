# FEA-FLUID-EXECUTION-001: Governed fluid execution and semantic publication

0 PASS, 2 PARTIAL, 0 FAIL, 36 OPEN, 0 TAGGED. Generated; do not edit.

## resources

Evidence kind: ResourceAdmission. Criterion: Admit original projection, source/initial realization, pressure/momentum operators, solver work, retained histories/outputs and transport before execution; cancellation and exact ceilings reject safely. Measured whole-process/backend usage must support claimed resource envelopes; requested bytes are not RSS.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | canonical_tetrahedra | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::execution::fluid::tests::authored_resource_ceiling_and_original_wire_profile_are_enforced_before_expansion` (crates/runmat-analysis/fea/src/execution/fluid/tests.rs:483) - Original payload and authored requested ceilings before expansion; no measured RSS/GPU/distributed bound. |
| incompressible_laminar_transient | canonical_tetrahedra | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | durable | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | durable | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | remote_execution | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | remote_execution | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | parallel_cpu | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | domain_decomposed | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | parallel_cpu | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | domain_decomposed | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | claimed_accelerator | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | claimed_accelerator | OPEN |  |

## placement

Evidence kind: ExecutionEquivalence. Criterion: Implement partitioned cell assembly after verified serial equations, then distributed linear algebra. Preserve original boundary/source flow and momentum resultants and numerical fields across serial/native/durable/encrypted-remote/partitioned/claimed-accelerator execution with supported precision and authenticated lifecycle.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | canonical_tetrahedra | numerical_kernel | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | numerical_kernel | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | durable | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | durable | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | remote_execution | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | remote_execution | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | parallel_cpu | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | domain_decomposed | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | parallel_cpu | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | domain_decomposed | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | claimed_accelerator | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | claimed_accelerator | OPEN |  |

## publication

Evidence kind: PublicationIntegrity. Criterion: Publish original complete state, observations, receipts and mandatory accuracy evidence atomically under exact source/phase lineage. Failed or partial imports cannot install completed authority. Include supported-envelope documentation and reproducible release evidence; no implicit channel fallback.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| incompressible_laminar_steady | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | public_study | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_transient | canonical_tetrahedra | native_process | OPEN |  |
| incompressible_laminar_steady | canonical_tetrahedra | durable | PARTIAL | `runmat-execution-runner-native[lib]::analysis_kernel::fluid_steady::tests::chain::publication::original_native_fluid_observation_complete_state_and_accuracy_cold_chain` (crates/runmat-execution-runner-native/src/analysis_kernel/fluid_steady/tests/chain/publication.rs:56) - Native durable complete-state/observation chain rejects missing mandatory spatial evidence; not a qualified convergence certificate. |
| incompressible_laminar_transient | canonical_tetrahedra | durable | OPEN |  |

