# FEA-SHARED-SEQUENCES-001: Semantic procedure transitions and immutable completed state

0 PASS, 4 PARTIAL, 0 FAIL, 44 OPEN, 0 TAGGED. Generated; do not edit.

## typed_semantic_graph

Evidence kind: SemanticRealization. Criterion: An engineering study is an ordered semantic sequence of procedures, not merely one procedure plus an execution task graph. `runmat-analysis-core` owns the authored sequence. `runmat-analysis-fea` validates procedure compatibility, realizes state transitions, and records their evidence. The generic execution graph schedules that already validated sequence; it cannot invent physical dependencies or ...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::execution::tests::semantic_state_and_reuse_transition_bind_exact_content_and_topology` (crates/runmat-analysis/fea/src/execution/tests.rs:304) - Semantic state and reuse transitions bind exact content and topology.<br>`runmat-analysis-fea[lib]::execution::tests::semantic_state_rejects_unimplemented_projection_transitions` (crates/runmat-analysis/fea/src/execution/tests.rs:493) - Semantic state rejects unimplemented projection transitions. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## distinct_completed_state_kinds

Evidence kind: SemanticRealization. Criterion: velocities, accelerations, and time-integrator state; temperature and thermal reference state; stress, strain, geometric-stiffness, and material-history state; contact active-set and friction history where the receiving procedure permits it; fluid state and moving-mesh/ALE state; electromagnetic complex or static field state; modal bases, normalization, participation, and truncation metadata; and ...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::execution::state::thermal::tests::completed_thermal_state_binds_authored_static_thermoelastic_reuse` (crates/runmat-analysis/fea/src/execution/state/thermal.rs:508) - A completed thermal state binds authored static thermoelastic reuse; structural and fluid state kinds are separate. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## activation_and_time_domains

Evidence kind: SemanticRealization. Criterion: Loads, sources, constraints, contacts, interfaces, initial conditions, output requests, and supported material parameters declare step-scoped activation. Activation is a typed lifecycle of `Activate`, `Continue`, `Modify`, or `Deactivate`; absence is not interpreted as an implicit continuation. A modification produces a new authored identity while retaining attribution to the original object. Ever...

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

## compatible_physical_transitions

Evidence kind: SemanticRealization. Criterion: Initial production transitions include: static structural state to prestressed modal extraction; static or supported nonlinear structural state to eigenvalue buckling; static or supported nonlinear structural state to prestressed direct or modal harmonic response; thermal state to structural temperature, thermal strain, and prestress; completed transient state to a continuation or restart of the s...

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

## complete_immutable_state_publication

Evidence kind: SemanticRealization. Criterion: Every state-producing step publishes an immutable, typed state manifest before dependent steps are admitted. The manifest includes producer-step identity, topology and partition identities, physical time or load parameter, field and history digests, reference conventions, linearization point, constraint plan, precision, and completeness evidence. A restart resumes an interrupted procedure from a c...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::execution::state::thermal::tests::completed_thermal_result_publishes_immutable_topology_bound_terminal_state` (crates/runmat-analysis/fea/src/execution/state/thermal.rs:310) - A completed thermal result publishes an immutable topology-bound terminal state. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## governed_transition_corpus

Evidence kind: SemanticRealization. Criterion: Required governed cases include: static preload to prestressed modal frequency shift; base-state stress to eigenvalue buckling load factors and modes; thermal solution to thermal strain, reaction, and prestressed response; transient continuation matching an uninterrupted reference run; branching one base state into modal, harmonic, and buckling procedures without mutation; correct activation, modi...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-core[lib]::validate_sequence::tests::rejects_upstream_input_without_governed_transition` (crates/runmat-analysis/core/src/validate_sequence/tests.rs:316) - Upstream input without a governed transition is rejected; the preload-to-prestressed-modal case is covered by study planning tests. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

