use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Rayleigh damping is proportional to the mass and stiffness matrices. The modal damping ratio at angular frequency omega is zeta = alpha / (2 omega) + beta omega / 2, so alpha controls low-frequency damping and beta high-frequency damping. To hit a ratio zeta at two frequencies omega1 and omega2, alpha = 2 zeta omega1 omega2 / (omega1 + omega2) and beta = 2 zeta / (omega1 + omega2).",
            "`massCoefficient` must be a `fea.RayleighMassCoefficient` in 1/s and `stiffnessCoefficient` a `fea.RayleighStiffnessCoefficient` in s. At least one must be positive: a pair of zeros is rejected (use `fea.MechanicalDamping.undamped()` instead). The coefficient objects are stored as `mass_coefficient` and `stiffness_coefficient`.",
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
    id: "rayleigh",
    title: "Create a Rayleigh (proportional) damping law",
    program: "alpha = fea.RayleighMassCoefficient.perSecond(2.5);\nbeta = fea.RayleighStiffnessCoefficient.seconds(1.0e-5);\ndamping = fea.MechanicalDamping.rayleigh(alpha, beta);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(damping), 'fea.MechanicalDamping'));\nassert(strcmp(damping.kind, 'rayleigh'));\nassert(damping.mass_coefficient.per_second == 2.5);\nassert(damping.stiffness_coefficient.seconds == 1.0e-5);",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "How do I get 2 percent damping between 10 and 100 Hz?",
        answer: "With omega1 = 2 pi 10 and omega2 = 2 pi 100 rad/s, alpha = 2(0.02) omega1 omega2/(omega1 + omega2), about 2.28 1/s, and beta = 2(0.02)/(omega1 + omega2), about 5.79e-5 s.",
    },
];

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
        title: "fea.MechanicalDamping.rayleigh",
        summary: "Create a Rayleigh (proportional) damping law.",
        description: "`fea.MechanicalDamping.rayleigh(massCoefficient, stiffnessCoefficient)` returns a `fea.MechanicalDamping` with damping matrix C = alpha M + beta K.",
        keywords: &[
            "fea",
            "damping",
            "Rayleigh",
            "proportional damping",
            "material",
        ],
        related: &[
            "fea.MechanicalDamping.undamped",
            "fea.material",
            "fea.RayleighMassCoefficient.perSecond",
            "fea.RayleighStiffnessCoefficient.seconds",
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
