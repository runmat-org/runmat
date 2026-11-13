use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "The reference temperature is the constitutive reference state of the thermal formulation, the temperature at which temperature-dependent properties are referenced.",
            "`referenceTemperature` must be a `fea.Temperature` (stored in kelvin); `fea.Temperature.celsius(20)` stores 293.15 K. Any other class, including `fea.Frequency` or `fea.TemperatureDifference`, is rejected. The value is stored as `reference_temperature`.",
            "All inputs are checked before anything is built: a wrong argument count, a request for more than one output, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the domain to `fea.model` in the `Domains` option (`'Domains', {domain}`). A model accepts at most one domain of each family; a second domain of the same family is rejected.",
            "Use it with `fea.ThermalStep.steady` or `fea.ThermalStep.transient`.",
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
    id: "atReferenceTemperature",
    title: "Create a standalone thermal domain with a reference temperature",
    program: "domain = fea.ThermalDomain.atReferenceTemperature(fea.Temperature.celsius(20));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(domain), 'fea.ThermalDomain'));\nassert(abs(domain.reference_temperature.kelvin - 293.15) < 1e-9);",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Is the reference temperature the initial temperature?",
        answer: "No. It is the constitutive reference state. The initial temperature field of a transient analysis is set separately with `fea.ThermalInitialCondition.temperature`.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "FEA models",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/models"),
    },
    BuiltinDocumentationLink {
        label: "Thermal physics",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/physics/thermal"),
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
        title: "fea.ThermalDomain.atReferenceTemperature",
        summary: "Create a standalone thermal domain with a reference temperature.",
        description: "`fea.ThermalDomain.atReferenceTemperature(referenceTemperature)` builds a `fea.ThermalDomain` that sets the reference temperature of a heat-transfer analysis.",
        keywords: &[
            "fea",
            "domain",
            "reference temperature",
        ],
        related: &[
            "fea.Temperature.celsius",
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
