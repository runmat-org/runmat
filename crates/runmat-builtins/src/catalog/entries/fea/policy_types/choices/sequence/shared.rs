//! Documentation facts shared by every identity of this policy family.

use crate::fea_contract::policy_types::documentation::{page, section, source, unit_test};
use crate::*;

pub(super) const USE_STATE_OUTPUT: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "Pass the value to `fea.StateOutput.new(id, kind, completeness, persistence)` to declare state a step publishes, or as the kind argument of `fea.StateInput.initialConditions(id, kind, useAs, conflict)` and `fea.StateInput.upstream(id, kind, producerStepId, stateId, useAs, conflict)` to declare state a step consumes. Outputs and inputs are attached with `fea.ProcedureStep.configure` and the steps are ordered by `fea.AnalysisSequence.new`.",
        "Sequence validation checks that the step's physics family can publish or consume the kind, and that an upstream input names a state of the same kind published by its producer.",
    ],
);
pub(super) const USE_STATE_INPUT: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "Pass the value to `fea.StateInput.initialConditions(id, kind, useAs, conflict)` or `fea.StateInput.upstream(id, kind, producerStepId, stateId, useAs, conflict)`, and attach the input with `fea.ProcedureStep.configure`. A transfer between steps also needs a `fea.SequenceDependency` with reason `stateTransfer` and a `fea.StateTransition`.",
        "Executable support is narrower than what validates: structural execution accepts at most one upstream input, either a structural-primary initial guess or a solved thermal state as thermomechanical prestress; modal-reduced steps consume a modal basis from their extraction step.",
    ],
);
pub(super) const USE_DEPENDENCY: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "Pass the reason to `fea.SequenceDependency.new(predecessorStepId, successorStepId, reason)` and list the dependency in `fea.AnalysisSequence.new(id, steps, dependencies, transitions, branches)`. Every reason orders the two steps; the reason records why.",
    ],
);
pub(super) const USE_TRANSITION: BuiltinDocumentationSection = section(
    "Use in a study",
    &[
        "Pass the policy to `fea.StateTransition.new(id, producerStepId, consumerStepId, stateId, method, topology, reference)` and list the transition in `fea.AnalysisSequence.new`. The transition needs a direct `stateTransfer` dependency between the two steps and a complete, persisted producer state.",
        "Validation enforces the combinations: `reuse` requires identical topology, adapted-mesh lineage requires conservative projection or interpolation, `initialGuessOnly` pairs with an `initialGuess` state use, and `linearize` or `deriveOperatorContribution` require an explicit linearization point. Execution today performs only byte-preserving `reuse` or `initialGuessOnly` transfers on identical topology with the reference state preserved; other validated transitions are rejected when the study runs.",
    ],
);
pub(super) const LINKS: &[BuiltinDocumentationLink] = &[
    page(
        "Procedure sequences, observations, and accuracy",
        "/docs/simulation/study-contracts",
    ),
    page(
        "Studies, analyses, and state",
        "/docs/simulation/language/studies-analyses-and-state",
    ),
];
pub(super) const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[
    source(
        "Sequence policy constructors",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/study_policies/sequence.rs",
    ),
    source(
        "Sequence state validation",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-analysis/core/src/validate_sequence/state.rs",
    ),
    source(
        "Step state validation",
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-analysis/core/src/validate_sequence/step.rs",
    ),
];
pub(super) const VERIFICATION: &[BuiltinEvidenceReference] = &[
    unit_test(
        "Configured sequence round trip",
        "crates/runmat-runtime/src/builtins/fea/sequences.rs::tests::configured_sequence_round_trips_with_typed_study_contracts",
    ),
    unit_test(
        "Sequence choices return typed catalog values",
        "crates/runmat-runtime/src/builtins/fea/study_policies/sequence/tests.rs::sequence_choices_return_typed_catalog_values",
    ),
    unit_test(
        "Catalog contract conformance",
        "crates/runmat-builtins/src/catalog/entries/fea/policy_types/choices/sequence/tests.rs::sequence_policy_choices_conform",
    ),
];
