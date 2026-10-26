use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Both ids are non-empty text.",
        "Model validation requires the producer to exist and be upstream, to publish `stateId` as a complete persisted state, and the step to declare that state as an upstream `fea.StateInput` with `fea.StateUse.restart()`.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the policy as the `restart` argument (8th) of `fea.ProcedureStep.configure`.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when an id is empty.",
        "`RunMat:fea:Internal` if the value cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "continue",
        title: "Continue loading from a converged preload",
        program: "policy = fea.RestartPolicy.continueFromCompletedState(\"preload\", \"converged_state\");",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(policy), 'fea.RestartPolicy'));\nassert(strcmp(policy.value.continue_from_completed_state.producer_step, 'preload'));\nassert(strcmp(policy.value.continue_from_completed_state.state_id, 'converged_state'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.RestartPolicy.continueFromCompletedState",
        summary: "Continue a step from a complete state of an upstream step.",
        description: "`fea.RestartPolicy.continueFromCompletedState(producerStepId, stateId)` builds a `fea.RestartPolicy` that continues a step from the complete state `stateId` published by `producerStepId`.",
        keywords: &["fea", "sequence", "restart", "continue", "state"],
        related: &["fea.StateInput.upstream", "fea.RestartPolicy.resumeCheckpoint", "fea.ProcedureStep.configure"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Do I also need a state input?", answer: "Yes. Declare the same producer state with `fea.StateInput.upstream(..., fea.StateUse.restart(), ...)`." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Studies, analyses, and state", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/studies-analyses-and-state") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/sequences.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/restart_policy/tests.rs::restart_policy_constructors_follow_their_positional_contract" },
        ],
        notes: &[],
    });
