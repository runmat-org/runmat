use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`maximumCount` must be a nonnegative integer scalar; integer classes are accepted exactly. Model validation requires it to be nonzero.",
        "Pair it with an explicit sampling schedule such as `fea.SamplingPolicy.timeSeconds`; the bound lets resource admission size storage before the solve starts. Each retained sample is an independently addressable chunk.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass it to `fea.ObservationStorage.new(precision, compression, retention, partialAccess)`; the storage object is the 13th argument of `fea.Observation.field`, `history`, `probe`, and `reduction`.",
        "`fea.Observation.diagnostic` takes a retention policy directly as its fourth argument.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when `maximumCount` is not a nonnegative integer scalar.",
        "`RunMat:fea:Internal` if the value cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "three-times",
        title: "Retain three scheduled samples",
        program: "value = fea.RetentionPolicy.explicitSamples(3);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(value), 'fea.RetentionPolicy'));\nassert(value.value.explicit_samples.maximum_count == 3);" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.RetentionPolicy.explicitSamples",
        summary: "Keep explicitly requested samples up to a declared maximum.",
        description: "`fea.RetentionPolicy.explicitSamples(maximumCount)` builds a bounded `fea.RetentionPolicy` that keeps the explicitly scheduled samples and declares their upper bound.",
        keywords: &["fea", "observation", "retention", "bounded", "explicit", "samples"],
        related: &["fea.RetentionPolicy.latestSamples", "fea.SamplingPolicy.timeSeconds", "fea.ObservationStorage.new"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "How does this map to `.fea` source?", answer: "It is `retention = explicit_samples` with `maximum_samples = N` in an output block." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/storage.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/retention_policy/tests.rs::retention_policy_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Bounded histories end to end", location: "crates/runmat-analysis/language/tests/output_histories.rs::bounded_sampling_interpolation_and_series_reductions_are_typed_end_to_end" },
        ],
        notes: &[],
    });
