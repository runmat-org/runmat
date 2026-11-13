use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Takes no arguments and returns an immutable host value.",
        "Use it for complex results from harmonic, acoustic, or electromagnetic frequency-domain steps. Model validation requires the precision to be complex exactly when the observation's complex representation is `fea.ComplexRepresentation.complexValue()`.",
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
        id: "complex64",
        title: "Select complex64",
        program: "value = fea.SignalPrecision.complex64();",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(value), 'fea.SignalPrecision'));\nassert(strcmp(value.value, 'complex64'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.SignalPrecision.complex64",
        summary: "Store complex observation samples as two 32-bit parts.",
        description: "`fea.SignalPrecision.complex64()` returns the typed choice `complex64` of `fea.SignalPrecision`.",
        keywords: &["fea", "observation", "precision", "complex", "harmonic"],
        related: &["fea.SignalPrecision.complex128", "fea.ObservationStorage.new"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Can I store a magnitude with a complex precision?", answer: "No. Magnitude and phase views are real, so they need `real32` or `real64`." }],
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
