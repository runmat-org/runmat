use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "The frame has no parameters and no properties. At each boundary point the first component is along the oriented face normal and the second and third are two tangents completing a right-handed set, so it follows curved faces. `fea.StructuralConstraint.normalDisplacement` uses this frame.",
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
    id: "boundaryNormalTangent",
    title: "Select the local boundary normal-tangent frame",
    program: "frame = fea.CoordinateFrame.boundaryNormalTangent();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(frame), 'fea.BoundaryNormalTangentFrame'));",
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
        title: "fea.CoordinateFrame.boundaryNormalTangent",
        summary: "Select the local boundary normal-tangent frame.",
        description: "`fea.CoordinateFrame.boundaryNormalTangent()` returns a `fea.BoundaryNormalTangentFrame` whose axes are the local outward normal and two tangents of each boundary face.",
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
