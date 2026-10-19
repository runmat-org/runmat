use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`maximumRelativeError` must be a finite real scalar; integer classes are accepted. The constructor stores it as given.",
        "Model validation requires 0 < `maximumRelativeError` < 1, and rejects error-bounded compression on observations whose purpose is `solverControl` or `downstreamState`, because lossy values must not steer a solve or seed another step.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass it to `fea.ObservationStorage.new(precision, compression, retention, partialAccess)`; the storage object is the 13th argument of `fea.Observation.field`, `history`, `probe`, and `reduction`.",
        "Use it for large visualization or publication fields where a documented error bound is acceptable.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when the bound is not a finite real numeric scalar.",
        "`RunMat:fea:Internal` if the value cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "visual",
        title: "Bound compression error at 0.01 %",
        program: "value = fea.CompressionPolicy.errorBounded(1e-4);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(value), 'fea.CompressionPolicy'));\nassert(value.value.error_bounded.maximum_relative_error == 1e-4);" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.CompressionPolicy.errorBounded",
        summary: "Compress stored observation samples within a relative error bound.",
        description: "`fea.CompressionPolicy.errorBounded(maximumRelativeError)` builds a lossy `fea.CompressionPolicy` whose decompressed values stay within the given relative error.",
        keywords: &["fea", "observation", "compression", "lossy", "error", "bound"],
        related: &["fea.CompressionPolicy.lossless", "fea.ObservationStorage.new"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Is the bound a percentage?", answer: "No. It is a fraction: 1e-4 means 0.01 %." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/storage.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/compression_policy/tests.rs::compression_policy_constructors_follow_their_positional_contract" },
        ],
        notes: &[],
    });
