use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "The frame has no parameters and no properties. It names element-local axes. Structural constraint and load evaluation does not support this frame yet: a `fea.StructuralConstraint.prescribed` that uses it constructs, but the structural solve rejects it as unsupported for pointwise geometry.",
            "All inputs are checked before anything is built: a wrong argument count, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the frame to `fea.StructuralConstraint.prescribed` as its `frame` argument; the DOF axes ux..rz are then the axes of this frame.",
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
    id: "element",
    title: "Select the element coordinate frame",
    program: "frame = fea.CoordinateFrame.element();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(frame), 'fea.ElementCoordinateFrame'));",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[];

const LINKS: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Structural constraints",
    target: BuiltinDocumentationLinkTarget::Documentation(
        "/docs/simulation/constraints/structural",
    ),
}];

const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/constraints/support/frame.rs"),
}];

const VERIFICATION: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Coordinate frames validate and round-trip",
        location: "crates/runmat-runtime/src/builtins/fea/constraints.rs::tests::coordinate_frames_validate_and_round_trip_without_string_dispatch",
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    documentation(ConstructorDocumentationDefinition {
        title: "fea.CoordinateFrame.element",
        summary: "Select the element coordinate frame.",
        description: "`fea.CoordinateFrame.element()` returns a `fea.ElementCoordinateFrame` whose axes are each element's own local axes.",
        keywords: &[
            "fea",
            "coordinate frame",
            "axes",
        ],
        related: &[
            "fea.StructuralConstraint.prescribed",
            "fea.CoordinateFrame.cartesian",
            "fea.CoordinateFrame.cylindrical",
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
