# FEA-RELEASE-COMPLETION-001: Per-formulation complete acceptance

1 PASS, 0 FAIL, 23 OPEN, 0 TAGGED. Generated; do not edit.

## exact_admitted_physics_and_upstream_state

Evidence kind: PublicationIntegrity. Criterion: A formulation is complete only when all of the following are true: Its public contract names the actual equations, procedure, assumptions, and supported envelope. Its identity is constructed only from a qualified pre-solve admission receipt and records exact spatial, temporal, constitutive, coupling, recovery, topology, source, and boundary realizations; identity validity is rechecked at persisten...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## truthful_fields_operators_and_accuracy

Evidence kind: PublicationIntegrity. Criterion: Its discrete formulation is implemented in one analysis-owned module hierarchy. All fields have correct units, locations, topology, and recovery semantics. Field, history, probe, reduction, monitor, and diagnostic requests have correct quantities, targets, locations, frames, sampling, retention, and cost admission. Result-field selection and observation evaluation remain separate from ordinary `pl...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PASS | `runmat-analysis-fea[lib]::results::accuracy::tests::production_accuracy_failure_produces_a_blocking_gate` (crates/runmat-analysis/fea/src/results/accuracy/tests.rs:32) - Scalar residual criterion failure remains blocking through a typed accuracy artifact; no complete governed family corpus, release attestation or public receipt ... |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## supported_execution_resources_and_release_evidence

Evidence kind: PublicationIntegrity. Criterion: Signal interpolation and excitation realization remain independent of task partitioning, worker count, and placement within the declared numerical-equivalence policy. Large numerical inputs, outputs, and checkpoints are chunked content-addressed artifacts with partial access; they are not gathered through the driver. Partitioned observations and reductions preserve values, locations, attribution, ...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

