use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Takes no arguments and returns an immutable host value.",
        "Samples are retained with the published study result. When the observation declares an upper-bound cost, model validation requires a nonzero stored-bytes estimate.",
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
        id: "canonical",
        title: "Select canonical",
        program: "value = fea.RetentionPolicy.canonical();",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(value), 'fea.RetentionPolicy'));\nassert(strcmp(value.value, 'canonical'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.RetentionPolicy.canonical",
        summary: "Keep every observation sample as part of the canonical result.",
        description: "`fea.RetentionPolicy.canonical()` returns the typed choice `canonical` of `fea.RetentionPolicy`.",
        keywords: &["fea", "observation", "retention", "canonical", "storage"],
        related: &["fea.RetentionPolicy.untilDependentsComplete", "fea.ObservationStorage.new"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Is canonical retention required for series reductions?", answer: "Series reductions need every produced sample, so use `canonical` or `untilDependentsComplete`, not `latestSamples`." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/storage.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/retention_policy/tests.rs::retention_policy_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed sequence round trip", location: "crates/runmat-runtime/src/builtins/fea/sequences.rs::tests::configured_sequence_round_trips_with_typed_study_contracts" },
        ],
        notes: &[],
    });
