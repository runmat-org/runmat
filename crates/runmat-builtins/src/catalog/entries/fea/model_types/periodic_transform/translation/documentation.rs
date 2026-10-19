use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`translation` must be a `fea.Translation` vector, for example `fea.Translation.meters([0.2 0 0])` for a unit cell 0.2 m long in x. The vector points from the primary region to the secondary region and is stored in meters.",
            "All inputs are checked before anything is built: a wrong argument count, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass it as the `transform` argument of a periodic constraint (`fea.StructuralConstraint.periodic`, `fea.ThermalConstraint.affineJump`, and the other families).",
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
    id: "translation",
    title: "Create a translational periodic transform",
    program: "transform = fea.PeriodicTransform.translation(fea.Translation.meters([0.2 0 0]));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(transform), 'fea.TranslationPeriodicTransform'));\nassert(strcmp(class(transform.translation), 'fea.Translation'));",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Which way does the translation point?",
        answer: "From the primary region to the secondary region. If you swap the regions, negate the translation.",
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
        title: "fea.PeriodicTransform.translation",
        summary: "Create a translational periodic transform.",
        description: "`fea.PeriodicTransform.translation(translation)` returns a `fea.TranslationPeriodicTransform` that maps a primary point x to x + t on the secondary region.",
        keywords: &[
            "fea",
            "periodic",
            "transform",
            "translation",
            "unit cell",
        ],
        related: &[
            "fea.PeriodicTransform.rotation",
            "fea.Translation.meters",
            "fea.StructuralConstraint.periodic",
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
