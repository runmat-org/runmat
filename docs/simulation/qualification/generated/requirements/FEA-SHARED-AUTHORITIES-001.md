# FEA-SHARED-AUTHORITIES-001: One shared topology, numerical and property authority

0 PASS, 1 PARTIAL, 0 FAIL, 23 OPEN, 0 TAGGED. Generated; do not edit.

## shared_topology_property_and_constraint_preparation

Evidence kind: SemanticRealization. Criterion: Implement and test the shared analysis projection. Implement one canonical property compiler from authored, geometry-resolved material/composite, section, layup, and orientation intent to an element-complete compiled property map. Bind its semantic digest to geometry revision and topology, keep audit evidence separate, and expose role/capability-specific immutable views rather than family-private ...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::constraints::compilation::preparation::tests::preparation_composes_shared_inventories_and_preserves_original_dispatch` (crates/runmat-analysis/fea/src/constraints/compilation/preparation/tests.rs:16) - Preparation composes shared inventories and preserves original dispatch. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## canonical_compute_and_restored_domain_owners

Evidence kind: SemanticRealization. Criterion: Move flow topology from runtime into analysis. Move acoustic, CFD, CHT, and FSI kernels out of runtime. Split runtime operations, documents, persistence, presentation, and policy. Split FEA topology, algebra, constraints, excitation, physics, orchestration, execution, results, and diagnostics. Add architecture tests for dependency direction, prohibited physics ownership in runtime, and prohibited ...

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

## single_authority_exit_invariants

Evidence kind: SemanticRealization. Criterion: Empty, conflicting, redundant, overconstrained, and underconstrained plans produce attributable diagnostics. Every promoted solver consumes the same canonical operator contract and rejects algebraically incompatible algorithms. Runtime sparse values, execution sparse values, and analysis operators have explicit one-way adapters and no ownership leakage. Field recovery, observation evaluation, plot...

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

