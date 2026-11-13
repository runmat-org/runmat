use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "An undamped law adds no damping matrix. Declaring it makes the choice explicit in the material and in run evidence, rather than leaving damping unspecified. It has no coefficients.",
            "All inputs are checked before anything is built: a wrong argument count, a request for more than one output, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the law to `fea.material` with the `Damping` option, for example `fea.material('steel', 'YoungsModulusPa', 200e9, 'PoissonRatio', 0.3, 'Damping', damping)`. The material then carries the damping into every region it is assigned to.",
            "Dynamic steps (harmonic and transient) use the material damping; static steps ignore it.",
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
    id: "undamped",
    title: "Declare a material explicitly undamped",
    program: "damping = fea.MechanicalDamping.undamped();\nsteel = fea.material('steel', 'YoungsModulusPa', 200e9, 'PoissonRatio', 0.3, 'Damping', damping);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(damping), 'fea.MechanicalDamping'));\nassert(strcmp(damping.kind, 'undamped'));\nassert(strcmp(class(steel), 'fea.Material'));",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[];

const LINKS: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "FEA models",
    target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/models"),
}];

const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/damping.rs"),
}];

const VERIFICATION: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Typed damping laws are portable and reject a zero Rayleigh pair",
        location: "crates/runmat-runtime/src/builtins/fea/damping.rs::tests::typed_damping_laws_are_portable_and_reject_a_zero_rayleigh_pair",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "fea.material preserves typed Rayleigh damping",
        location: "crates/runmat-runtime/src/builtins/fea/mod.rs::tests::fea_material_preserves_typed_rayleigh_damping",
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    documentation(ConstructorDocumentationDefinition {
        title: "fea.MechanicalDamping.undamped",
        summary: "Declare a material explicitly undamped.",
        description: "`fea.MechanicalDamping.undamped()` returns a `fea.MechanicalDamping` with `kind` `'undamped'`.",
        keywords: &[
            "fea",
            "damping",
            "material",
            "undamped",
        ],
        related: &[
            "fea.MechanicalDamping.rayleigh",
            "fea.material",
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
