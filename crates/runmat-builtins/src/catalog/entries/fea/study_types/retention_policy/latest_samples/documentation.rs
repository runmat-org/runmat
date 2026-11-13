use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`count` must be a nonnegative integer scalar; integer classes and integer-valued doubles are accepted exactly. Model validation requires it to be nonzero.",
        "Older samples are dropped, so a history kept this way cannot be series-reduced: RunMat rejects a statistic over an accidental suffix. Long-running histories still publish checkpoints that bind the full sample count and a rolling digest of every observed sample.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass it to `fea.ObservationStorage.new(precision, compression, retention, partialAccess)`; the storage object is the 13th argument of `fea.Observation.field`, `history`, `probe`, and `reduction`.",
        "`fea.Observation.diagnostic` takes a retention policy directly as its fourth argument.",
        "It suits convergence diagnostics and progress monitors where only recent values matter.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when `count` is not a nonnegative integer scalar.",
        "`RunMat:fea:Internal` if the value cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "rolling",
        title: "Keep the last 100 convergence samples",
        program: "value = fea.RetentionPolicy.latestSamples(100);\ndiagnostic = fea.Observation.diagnostic(\"convergence\", fea.DiagnosticKind.convergence(), ...\n    fea.SamplingPolicy.everyAcceptedIncrement(), value);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(value), 'fea.RetentionPolicy'));\nassert(value.value.latest_samples.count == 100);\nassert(strcmp(class(diagnostic), 'fea.Observation'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.RetentionPolicy.latestSamples",
        summary: "Keep only the most recent observation samples.",
        description: "`fea.RetentionPolicy.latestSamples(count)` builds a bounded `fea.RetentionPolicy` that keeps the latest `count` samples and discards older ones.",
        keywords: &["fea", "observation", "retention", "bounded", "latest", "rolling"],
        related: &["fea.RetentionPolicy.explicitSamples", "fea.Observation.diagnostic", "fea.ObservationStorage.new"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Is `latestSamples(0)` accepted?", answer: "The constructor accepts it, but model validation rejects a zero count." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/storage.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/retention_policy/tests.rs::retention_policy_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Closed output policy", location: "crates/runmat-analysis/language/tests/output_histories.rs::output_policy_is_closed_and_bounded" },
        ],
        notes: &[],
    });
