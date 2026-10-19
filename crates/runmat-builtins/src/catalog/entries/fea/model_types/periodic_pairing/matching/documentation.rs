use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Matching pairing expects every primary entity to have exactly one partner after the transform is applied; mesh validation reports a pairing failure if the meshes do not match. It has no properties.",
            "All inputs are checked before anything is built: a wrong argument count, a request for more than one output, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass it as the `pairing` argument of a tie, periodic, phase-shifted periodic, or affine-jump constraint.",
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
    id: "matching",
    title: "Require one-to-one matching entities on paired regions",
    program: "pairing = fea.PeriodicPairing.matching();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(pairing), 'fea.MatchingPeriodicPairing'));",
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
        title: "fea.PeriodicPairing.matching",
        summary: "Require one-to-one matching entities on paired regions.",
        description: "`fea.PeriodicPairing.matching()` returns a `fea.MatchingPeriodicPairing` that requires the two regions of a tie or periodic constraint to have matching meshes.",
        keywords: &[
            "fea",
            "periodic",
            "pairing",
            "matching mesh",
        ],
        related: &[
            "fea.PeriodicPairing.geometric",
            "fea.StructuralConstraint.tie",
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
