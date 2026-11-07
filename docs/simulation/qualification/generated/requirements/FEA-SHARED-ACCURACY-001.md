# FEA-SHARED-ACCURACY-001: Typed numerical accuracy acceptance and bounded refinement

0 PASS, 2 PARTIAL, 0 FAIL, 22 OPEN, 0 TAGGED. Generated; do not edit.

## dimensioned_targets_and_contributions

Evidence kind: AccuracyAcceptance. Criterion: A converged algebraic solve is not automatically an accurate engineering result. Every procedure step therefore owns a typed `AccuracyPlan` and publishes an `AccuracyReport`. These coordinate existing family-specific controls without pretending that unlike error measures can always be collapsed into one rigorous scalar bound. An authored accuracy plan may declare: target quantities or observations...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::adaptation::solved_source::tests::original_both_order_static_results_retain_typed_convergence_and_scalar_observations` (crates/runmat-analysis/fea/src/adaptation/solved_source/tests.rs:16) - Original static results retain typed convergence and scalar observations for both element orders. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## mandatory_acceptance_states

Evidence kind: AccuracyAcceptance. Criterion: Each contribution retains its own norm, units, estimator assumptions, scope, and provenance. The system does not add dimensionally or statistically incompatible values into a decorative “total error.” It instead evaluates a typed acceptance predicate, identifies the governing unmet criterion, and distinguishes: `Satisfied`: every mandatory criterion has passing evidence; `NotSatisfied`: at least o...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::results::accuracy::tests::production_accuracy_failure_produces_a_blocking_gate` (crates/runmat-analysis/fea/src/results/accuracy/tests.rs:32) - Scalar residual criterion failure remains blocking through a typed accuracy artifact; no complete governed family corpus, release attestation or public receipt ...<br>`runmat-analysis-fea[lib]::results::accuracy::tests::mandatory_evidence_governs_status_and_production_publication` (crates/runmat-analysis/fea/src/results/accuracy/tests.rs:66) - Mandatory evidence governs status and publication.<br>`runmat-analysis-fea[lib]::results::accuracy::tests::missing_mandatory_evidence_is_indeterminate_and_blocks_by_default` (crates/runmat-analysis/fea/src/results/accuracy/tests.rs:87) - Missing mandatory evidence is indeterminate and blocks publication by default. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## bounded_refinement_and_governed_corpus

Evidence kind: AccuracyAcceptance. Criterion: Target-quantity accuracy integrates with typed observations. A target must be computable on the current formulation and carry a valid recovery and sampling policy. Adaptive analysis consumes the same accuracy plan rather than defining a parallel convergence authority. Coupled studies retain both per-family contributions and interface/global closure; one passing component cannot hide another failin...

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

