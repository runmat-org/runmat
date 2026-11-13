use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Takes no arguments and returns an immutable host value.",
        "Keeps the full binary64 real and imaginary parts of frequency-domain results. Model validation requires the precision to be complex exactly when the observation's complex representation is `fea.ComplexRepresentation.complexValue()`.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass it to `fea.ObservationStorage.new(precision, compression, retention, partialAccess)`; the storage object is the 13th argument of `fea.Observation.field`, `history`, `probe`, and `reduction`.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:Internal` if the value cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "complex128",
        title: "Select complex128",
        program: "value = fea.SignalPrecision.complex128();",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(value), 'fea.SignalPrecision'));\nassert(strcmp(value.value, 'complex128'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.SignalPrecision.complex128",
        summary: "Store complex observation samples as two 64-bit parts.",
        description: "`fea.SignalPrecision.complex128()` returns the typed choice `complex128` of `fea.SignalPrecision`.",
        keywords: &["fea", "observation", "precision", "complex", "double"],
        related: &["fea.SignalPrecision.complex64", "fea.ObservationStorage.new"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "When do I need a complex precision?", answer: "When the observation keeps the complex value itself rather than a real, imaginary, magnitude, or phase view." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/storage.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/signal_precision/tests.rs::signal_precision_constructors_follow_their_positional_contract" },
        ],
        notes: &[],
    });
