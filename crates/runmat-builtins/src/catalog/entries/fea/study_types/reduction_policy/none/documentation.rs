use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Takes no arguments. The policy carries no reduction, so the object has no `value` property to read.",
        "This is the only reduction a `fea.Observation.field` request may carry; model validation rejects any `fea.ReductionPolicy.new(...)` on a field request, even one whose parts are all `none`.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass it as the 12th argument (`reduction`) of `fea.Observation.field`, `history`, or `probe`. A `fea.Observation.reduction` request needs `fea.ReductionPolicy.new` with a non-trivial spatial or series reduction instead.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:Internal` if the value cannot be converted into a RunMat object.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "field",
        title: "Keep a field unreduced",
        program: "policy = fea.ReductionPolicy.none();",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(policy), 'fea.ReductionPolicy'));\nassert(isobject(policy));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.ReductionPolicy.none",
        summary: "Keep an observation unreduced.",
        description: "`fea.ReductionPolicy.none()` builds a `fea.ReductionPolicy` that states the observation keeps its full spatial and series data.",
        keywords: &["fea", "observation", "reduction", "none", "field"],
        related: &["fea.ReductionPolicy.new", "fea.Observation.field"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Why can't I read `policy.value`?", answer: "An absent reduction is stored as no value, so the object has no `value` property." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/storage.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/reduction_policy/tests.rs::reduction_policy_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed sequence round trip", location: "crates/runmat-runtime/src/builtins/fea/sequences.rs::tests::configured_sequence_round_trips_with_typed_study_contracts" },
        ],
        notes: &["Known gap: the registered immutable-value schema requires a `value` field, so encoding a standalone `none` policy through the value codec (for example across a process boundary) is rejected. The policy nested inside an observation is unaffected."],
    });
