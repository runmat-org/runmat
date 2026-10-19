use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Takes no arguments and returns an immutable host value.",
        "Samples are retained while dependent steps may need them and can be released afterwards. When the observation declares an upper-bound cost, model validation requires a nonzero stored-bytes estimate.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass it to `fea.ObservationStorage.new(precision, compression, retention, partialAccess)`; the storage object is the 13th argument of `fea.Observation.field`, `history`, `probe`, and `reduction`.",
        "`fea.Observation.diagnostic` takes a retention policy directly as its fourth argument.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:Internal` if the value cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "untildependentscomplete",
        title: "Select until_dependents_complete",
        program: "value = fea.RetentionPolicy.untilDependentsComplete();",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(value), 'fea.RetentionPolicy'));\nassert(strcmp(value.value, 'until_dependents_complete'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.RetentionPolicy.untilDependentsComplete",
        summary: "Keep observation samples until downstream steps finish.",
        description: "`fea.RetentionPolicy.untilDependentsComplete()` returns the typed choice `until_dependents_complete` of `fea.RetentionPolicy`.",
        keywords: &["fea", "observation", "retention", "temporary", "dependents"],
        related: &["fea.RetentionPolicy.canonical", "fea.ObservationStorage.new"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "When is this useful?", answer: "For intermediate results that feed later steps but are not part of the published result." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/storage.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/retention_policy/tests.rs::retention_policy_constructors_follow_their_positional_contract" },
        ],
        notes: &[],
    });
