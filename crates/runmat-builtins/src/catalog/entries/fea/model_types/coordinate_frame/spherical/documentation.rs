use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "At each point the components are radial (from `origin` to the point), polar, and azimuthal (polarAxis x radial). `azimuthReference` must be orthogonal to `polarAxis`; the constructor rejects parallel or non-orthogonal directions with `RunMat:fea:InvalidInput` before building the frame. The frame is undefined at the origin. Use it for domes and spherical shells.",
            "`id` names the frame. `origin` must be a `fea.Position`; `polarAxis` and `azimuthReference` must be `fea.Direction` values. The frame is validated when it is built: parallel or degenerate axes are rejected. The inputs are stored as the properties `frame_id`, `origin`, `polar_axis`, and `azimuth_reference`.",
            "All inputs are checked before anything is built: a wrong argument count, a request for more than one output, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
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
    id: "spherical",
    title: "Create a spherical frame from an origin, polar axis, and azimuth reference",
    program: "frame = fea.CoordinateFrame.spherical('dome', fea.Position.meters([0.1 0.05 0]), fea.Direction.vector([0 0 1]), fea.Direction.vector([1 0 0]));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(frame), 'fea.SphericalCoordinateFrame'));\nassert(strcmp(frame.frame_id, 'dome'));\nassert(strcmp(class(frame.polar_axis), 'fea.Direction'));",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "What happens if the two directions are parallel?",
        answer: "The constructor rejects the frame with `RunMat:fea:InvalidInput`, because the second direction is needed to fix the orientation about the first. Cartesian and spherical frames also reject directions that are not orthogonal; a cylindrical radial reference only has to be nonparallel to the axis.",
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
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Frames build every class and reject non-orthogonal axes before construction",
        location: "crates/runmat-runtime/src/builtins/fea/constraints/coverage_tests.rs::frames_build_every_class_and_reject_non_orthogonal_axes_before_construction",
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    documentation(ConstructorDocumentationDefinition {
        title: "fea.CoordinateFrame.spherical",
        summary: "Create a spherical frame from an origin, polar axis, and azimuth reference.",
        description: "`fea.CoordinateFrame.spherical(id, origin, polarAxis, azimuthReference)` returns a `fea.SphericalCoordinateFrame` positioned at `origin` and oriented by two directions.",
        keywords: &[
            "fea",
            "coordinate frame",
            "spherical",
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
