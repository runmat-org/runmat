use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Takes no arguments. This is the usual policy; it imposes no extra validation.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the policy as the `restart` argument (8th) of `fea.ProcedureStep.configure`.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Errors",
        paragraphs: &[
            "`RunMat:fea:Internal` if the value cannot be converted into a RunMat object.",
            "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "fresh",
        title: "Start a step fresh",
        program: "policy = fea.RestartPolicy.fresh();",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(policy), 'fea.RestartPolicy'));\nassert(strcmp(policy.value, 'fresh'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.RestartPolicy.fresh",
        summary: "Start a procedure step from its declared state inputs.",
        description: "`fea.RestartPolicy.fresh()` builds the `fea.RestartPolicy` for a step that starts from its declared state inputs rather than continuing or resuming earlier work.",
        keywords: &["fea", "sequence", "restart", "fresh", "step"],
        related: &["fea.RestartPolicy.continueFromCompletedState", "fea.RestartPolicy.resumeCheckpoint", "fea.ProcedureStep.configure"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Does `fresh` ignore upstream states?", answer: "No. Declared `fea.StateInput` values are still consumed; `fresh` only means the step is not a continuation or checkpoint resume." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Studies, analyses, and state", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/studies-analyses-and-state") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/sequences.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/restart_policy/tests.rs::restart_policy_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed sequence round trip", location: "crates/runmat-runtime/src/builtins/fea/sequences.rs::tests::configured_sequence_round_trips_with_typed_study_contracts" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Compiled sequence execution", location: "crates/runmat-vm/tests/fea_study_contracts.rs::typed_study_sequence_compiles_and_executes_through_the_vm" },
        ],
        notes: &[],
    });
