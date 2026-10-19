use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`tolerance` must be a nonnegative `fea.Length`, for example `fea.Length.micrometers(5)`. Choose it larger than the geometric mismatch between the regions and well below the smallest element size, so no entity can pair with the wrong partner.",
            "All inputs are checked before anything is built: a wrong argument count, a request for more than one output, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass it as the `pairing` argument of a tie, periodic, phase-shifted periodic, or affine-jump constraint. It is the usual choice for ties between independently meshed parts.",
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
    id: "geometric",
    title: "Pair entities on two regions geometrically within a length tolerance",
    program: "pairing = fea.PeriodicPairing.geometric(fea.Length.millimeters(0.01));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(pairing), 'fea.GeometricPeriodicPairing'));\nassert(abs(pairing.tolerance.meters - 1e-5) < 1e-18);",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[];

const LINKS: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Constraints overview",
    target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/constraints"),
}];

const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/constraints/support/pairing.rs"),
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
        title: "fea.PeriodicPairing.geometric",
        summary: "Pair entities on two regions geometrically within a length tolerance.",
        description: "`fea.PeriodicPairing.geometric(tolerance)` returns a `fea.GeometricPeriodicPairing` that pairs entities whose transformed positions agree within a tolerance.",
        keywords: &[
            "fea",
            "periodic",
            "pairing",
            "tolerance",
            "nonconforming",
        ],
        related: &[
            "fea.PeriodicPairing.matching",
            "fea.Length.millimeters",
            "fea.StructuralConstraint.tie",
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
