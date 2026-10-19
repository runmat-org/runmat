use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Fields are represented as complex phasors at the reference frequency f, with angular frequency omega = 2 pi f. Eddy currents and skin depth depend on this frequency.",
            "`frequency` must be a `fea.Frequency`, which is positive by construction (for example `fea.Frequency.hertz(60)` for mains frequency). It is stored as `harmonic.reference_frequency`.",
            "All inputs are checked before anything is built: a wrong argument count, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the domain to `fea.model` in the `Domains` option (`'Domains', {domain}`). A model accepts at most one domain of each family; a second domain of the same family is rejected.",
            "Use it with `fea.ElectromagneticStep.harmonic`.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Errors",
        paragraphs: &[
            "`RunMat:fea:InvalidInput`: an argument has the wrong class or shape, an identifier is empty, or a value violates the constructor's range rules.",
            "`RunMat:fea:Internal`: the validated object could not be converted to a RunMat value (not expected in normal use).",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "atFrequency",
    title: "Create a time-harmonic electromagnetic domain at a reference frequency",
    program: "domain = fea.ElectromagneticDomain.atFrequency(fea.Frequency.hertz(60));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(domain), 'fea.ElectromagneticDomain'));\nassert(domain.harmonic.reference_frequency.hertz == 60);",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "FEA models",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/models"),
    },
    BuiltinDocumentationLink {
        label: "Electromagnetic physics",
        target: BuiltinDocumentationLinkTarget::Documentation(
            "/docs/simulation/physics/electromagnetic",
        ),
    },
];

const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/domains.rs"),
}];

const VERIFICATION: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Domain constructors preserve typed physics boundaries",
        location: "crates/runmat-runtime/src/builtins/fea/mod.rs::tests::fea_domain_constructors_preserve_typed_physics_boundaries",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Domain classes expose only their own constructors",
        location: "crates/runmat-runtime/src/builtins/fea/mod.rs::tests::fea_domain_classes_expose_only_their_own_constructors",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "A model rejects duplicate domain families",
        location: "crates/runmat-runtime/src/builtins/fea/mod.rs::tests::fea_model_rejects_duplicate_domain_families",
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    documentation(ConstructorDocumentationDefinition {
        title: "fea.ElectromagneticDomain.atFrequency",
        summary: "Create a time-harmonic electromagnetic domain at a reference frequency.",
        description: "`fea.ElectromagneticDomain.atFrequency(frequency)` builds a `fea.ElectromagneticDomain` for a frequency-domain (time-harmonic) electromagnetic analysis.",
        keywords: &[
            "fea",
            "domain",
            "electromagnetic",
            "frequency",
            "harmonic",
            "eddy current",
        ],
        related: &[
            "fea.ElectromagneticStep.harmonic",
            "fea.Frequency.hertz",
            "fea.model",
        ],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: FAQS,
        links: LINKS,
        implementation: IMPLEMENTATION,
        verification: VERIFICATION,
        notes: &[
        ],
    });
