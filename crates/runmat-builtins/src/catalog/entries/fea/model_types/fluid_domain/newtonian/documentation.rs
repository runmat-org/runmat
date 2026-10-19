use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "A Newtonian fluid has a shear stress proportional to the strain rate, with constant density rho and dynamic viscosity mu; the kinematic viscosity is mu / rho. Water at 20 degrees C has rho = 998.2 kg/m^3 and mu = 1.002e-3 Pa s; air has about 1.204 kg/m^3 and 1.81e-5 Pa s.",
            "`density` must be a `fea.MassDensity`, `dynamicViscosity` a `fea.DynamicViscosity`, and `turbulenceIntensity` a `fea.TurbulenceIntensity` (a ratio from 0 to 1, for example 0.05 for 5 percent). The quantity constructors enforce their ranges. The values are stored as `density`, `dynamic_viscosity`, and `turbulence_intensity`.",
            "All inputs are checked before anything is built: a wrong argument count, a request for more than one output, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the domain to `fea.model` in the `Domains` option (`'Domains', {domain}`). A model accepts at most one domain of each family; a second domain of the same family is rejected.",
            "Use it with `fea.FluidStep.steadyState`, `fea.FluidStep.transient`, or a coupled fluid step.",
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
    id: "newtonian",
    title: "Create a Newtonian fluid domain from density, viscosity, and turbulence intensity",
    program: "domain = fea.FluidDomain.newtonian(fea.MassDensity.kilogramsPerCubicMeter(998.2), fea.DynamicViscosity.pascalSeconds(1.002e-3), fea.TurbulenceIntensity.ratio(0.05));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(domain), 'fea.FluidDomain'));\nassert(domain.density.kilograms_per_cubic_meter == 998.2);\nassert(domain.turbulence_intensity.ratio == 0.05);",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Do I give dynamic or kinematic viscosity?",
        answer: "Dynamic viscosity in Pa s. Kinematic viscosity in m^2/s equals dynamic viscosity divided by density.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "FEA models",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/models"),
    },
    BuiltinDocumentationLink {
        label: "Fluid physics",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/physics/fluid"),
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
        title: "fea.FluidDomain.newtonian",
        summary: "Create a Newtonian fluid domain from density, viscosity, and turbulence intensity.",
        description: "`fea.FluidDomain.newtonian(density, dynamicViscosity, turbulenceIntensity)` builds a `fea.FluidDomain` with constant Newtonian fluid properties.",
        keywords: &[
            "fea",
            "domain",
            "fluid",
            "Newtonian",
            "CFD",
            "viscosity",
            "density",
        ],
        related: &[
            "fea.FluidStep.steadyState",
            "fea.MassDensity.kilogramsPerCubicMeter",
            "fea.DynamicViscosity.pascalSeconds",
            "fea.TurbulenceIntensity.ratio",
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
