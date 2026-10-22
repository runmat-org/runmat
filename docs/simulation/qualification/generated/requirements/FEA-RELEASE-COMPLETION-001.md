# FEA-RELEASE-COMPLETION-001: Per-formulation complete acceptance

0 PASS, 0 PARTIAL, 0 FAIL, 6 OPEN, 0 TAGGED. Generated; do not edit.

## exact_admitted_physics_and_upstream_state

Evidence kind: PublicationIntegrity. Criterion: A formulation is complete only when all of the following are true: Its public contract names the actual equations, procedure, assumptions, and supported envelope. Its identity is constructed only from a qualified pre-solve admission receipt and records exact spatial, temporal, constitutive, coupling, recovery, topology, source, and boundary realizations; identity validity is rechecked at persisten...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## truthful_fields_operators_and_accuracy

Evidence kind: PublicationIntegrity. Criterion: Its discrete formulation is implemented in one analysis-owned module hierarchy. All fields have correct units, locations, topology, and recovery semantics. Field, history, probe, reduction, monitor, and diagnostic requests have correct quantities, targets, locations, frames, sampling, retention, and cost admission. Result-field selection and observation evaluation remain separate from ordinary `pl...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

## supported_execution_resources_and_release_evidence

Evidence kind: PublicationIntegrity. Criterion: Signal interpolation and excitation realization remain independent of task partitioning, worker count, and placement within the declared numerical-equivalence policy. Large numerical inputs, outputs, and checkpoints are chunked content-addressed artifacts with partial access; they are not gathered through the driver. Partitioned observations and reductions preserve values, locations, attribution, ...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | release_pipeline | OPEN |  |
| shared_infrastructure | mesh_independent | source_check | OPEN |  |

