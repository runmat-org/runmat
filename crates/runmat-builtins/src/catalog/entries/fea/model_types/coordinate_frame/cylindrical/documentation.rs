use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "At each point the components are radial (perpendicular from the axis to the point), azimuthal (axis x radial), and axial (along `axis`). `axis` is the cylinder axis through `origin`. `radialReference` must not be parallel to the axis; the radial direction itself is computed from each point's position, so the frame is undefined on the axis. Use it for shafts, bores, and pressure vessels.",
            "`id` names the frame. `origin` must be a `fea.Position`; `axis` and `radialReference` must be `fea.Direction` values. The frame is validated when it is built: parallel or degenerate axes are rejected. The inputs are stored as the properties `frame_id`, `origin`, `axis`, and `radial_reference`.",
            "All inputs are checked before anything is built: a wrong argument count, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the frame to `fea.StructuralConstraint.prescribed` as its `frame` argument; the DOF axes ux..rz then refer to this frame's first, second, and third components.",
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
    id: "cylindrical",
    title: "Create a cylindrical frame from an origin, axis, and radial reference",
    program: "frame = fea.CoordinateFrame.cylindrical('shaft', fea.Position.meters([0 0.05 0.05]), fea.Direction.vector([1 0 0]), fea.Direction.vector([0 1 0]));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(frame), 'fea.CylindricalCoordinateFrame'));\nassert(strcmp(frame.frame_id, 'shaft'));\nassert(strcmp(class(frame.axis), 'fea.Direction'));",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "What happens if the two directions are parallel?",
        answer: "The constructor rejects the frame with `RunMat:fea:InvalidInput`, because the second direction is needed to fix the orientation about the first.",
    },
];

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
        title: "fea.CoordinateFrame.cylindrical",
        summary: "Create a cylindrical frame from an origin, axis, and radial reference.",
        description: "`fea.CoordinateFrame.cylindrical(id, origin, axis, radialReference)` returns a `fea.CylindricalCoordinateFrame` positioned at `origin` and oriented by two directions.",
        keywords: &[
            "fea",
            "coordinate frame",
            "cylindrical",
        ],
        related: &[
            "fea.StructuralConstraint.prescribed",
            "fea.Direction.vector",
            "fea.Position.meters",
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
