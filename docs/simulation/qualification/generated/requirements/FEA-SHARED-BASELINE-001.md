# FEA-SHARED-BASELINE-001: Exact pre-solve admission and truthful realization identity

0 PASS, 2 PARTIAL, 0 FAIL, 22 OPEN, 0 TAGGED. Generated; do not edit.

## pre_solve_formulation_receipts

Evidence kind: SemanticRealization. Criterion: Fix standalone thermal persistence classification. Fix backend/device provenance and typed fallback reporting. Correct misleading field names and semantics. Separate the admitted formulation from numerical quality and publication status. Centralize pre-assembly formulation admission. Resolve the typed family, procedure, solution space, model, and mesh through one registry into an opaque admitted f...

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

## exact_realization_and_typed_contracts

Evidence kind: SemanticRealization. Criterion: Replace the conflated profile/run-kind model with clean family and procedure contracts. A typed per-procedure-step execution plan selects backend and solver options; it does not duplicate the step's family/procedure/solution-space identity or derive solver identity from a model template. Split constraints, natural/Robin boundaries, connectors, interfaces, and gauges into their typed authored model...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::admission::structural::tests::modal_response_spectrum_has_a_distinct_admitted_temporal_realization` (crates/runmat-analysis/fea/src/admission/structural.rs:451) - Modal response spectrum has a distinct admitted temporal realization. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## decode_revalidation_and_proxy_rejection

Evidence kind: SemanticRealization. Criterion: Admission either returns a registry-issued exact formulation receipt or fails before solve; a forged, incompatible, or unregistered identity is rejected on decode and persistence. Numerical acceptance and publication are distinct. Physical-discretization receipts cannot contain inferred lattices, generated graphs, implicit channels, synthetic profiles/operators, DOF-adjacency recovery, pair-spring...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::admission::tests::incompatible_exact_axis_is_rejected_before_admission` (crates/runmat-analysis/fea/src/admission/tests.rs:319) - An incompatible exact axis is rejected before admission.<br>`runmat-analysis-fea[lib]::admission::tests::unsupported_identity_and_solution_space_are_typed_rejections` (crates/runmat-analysis/fea/src/admission/tests.rs:347) - Unsupported identity and solution space are typed rejections.<br>`runmat-analysis-fea[lib]::admission::tests::persisted_identity_is_revalidated_and_cannot_forge_production` (crates/runmat-analysis/fea/src/admission/tests.rs:383) - Thermal-transient identity cold decode revalidates and rejects fields outside the registry identity; no end-to-end publication proof. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

