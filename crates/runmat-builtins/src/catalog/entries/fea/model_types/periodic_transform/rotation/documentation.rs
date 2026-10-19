use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "The axis passes through `origin` (a `fea.Position`) along `axis` (a `fea.Direction`), and the rotation by `angle` (a `fea.Rotation`) follows the right-hand rule about the axis direction. For a sector of a part with N identical sectors the angle is 360/N degrees. The three inputs are stored as given.",
            "All inputs are checked before anything is built: a wrong argument count, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass it as the `transform` argument of a periodic or phase-shifted periodic constraint to model one sector of a cyclically symmetric part, such as a fan, a flange, or a motor.",
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
    id: "rotation",
    title: "Create a rotational (cyclic) periodic transform",
    program: "transform = fea.PeriodicTransform.rotation(fea.Position.meters([0 0 0]), fea.Direction.vector([0 0 1]), fea.Rotation.degrees(45));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(transform), 'fea.RotationPeriodicTransform'));\nassert(abs(transform.angle.radians - pi/4) < 1e-15);",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "What sign does the angle have?",
        answer: "Positive angles follow the right-hand rule about `axis`: counterclockwise when viewed from a point on the positive axis looking back toward the origin. Reversing `axis` reverses the sense.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Constraints overview",
    target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/constraints"),
}];

const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/constraints/support/periodic_transform.rs"),
}];

const VERIFICATION: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Paired tie, periodic, phase-shifted, and affine-jump relations are typed and dimensionally checked",
        location: "crates/runmat-runtime/src/builtins/fea/constraints.rs::tests::paired_relations_are_typed_transformed_and_dimensionally_sound",
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    documentation(ConstructorDocumentationDefinition {
        title: "fea.PeriodicTransform.rotation",
        summary: "Create a rotational (cyclic) periodic transform.",
        description: "`fea.PeriodicTransform.rotation(origin, axis, angle)` returns a `fea.RotationPeriodicTransform` that rotates a primary point about an axis onto the secondary region.",
        keywords: &[
            "fea",
            "periodic",
            "transform",
            "rotation",
            "cyclic symmetry",
            "sector",
        ],
        related: &[
            "fea.PeriodicTransform.translation",
            "fea.Direction.vector",
            "fea.Rotation.degrees",
            "fea.MagneticConstraint.phaseShiftedPeriodic",
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
