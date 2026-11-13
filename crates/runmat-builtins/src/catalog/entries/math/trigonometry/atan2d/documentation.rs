use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationLink, BuiltinDocumentationLinkTarget, BuiltinDocumentationSection,
    BuiltinDocumentationStatus, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Four-quadrant angle",
        paragraphs: &["`atan2d(Y,X)` returns the four-quadrant inverse tangent in degrees in the interval [-180,180]. Inputs must be real and broadcast-compatible. Signed zero follows the existing atan2 convention, including zero when both coordinates are mathematically zero.", "A single-precision operand selects single output; otherwise output is double. Scalars, arrays, empty shapes and N-D broadcasting retain the numeric shape."],
    },
    BuiltinDocumentationSection {
        heading: "Execution and current limits",
        paragraphs: &["Resident numeric inputs gather through their shared exact provider owner, compute on the host and restore to that same owner. No degree-specific provider hook, shader fusion or profitable placement estimate is claimed. Typed integers, logical values and characters are RunMat-only extensions with an exact binary64 admission boundary.", "Tables and timetables use the existing binary variable planner and publish only after all result columns are rooted. Sparse, tall and symbolic overloads are not yet implemented."],
    },
];

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "quadrants",
    title: "Angles of four points on the unit circle",
    program: "x = [1 0 -1 0]; y = [0 1 0 -1]; d = atan2d(y,x)",
    display_output: Some("d = [0 90 180 -90]"),
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: crate::BuiltinExampleFixture::None,
    requirements: crate::BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isequal(d,[0 90 180 -90]));",
    },
}];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("atan2d"), slug: Some("atan2d"),
    summary: "Compute four-quadrant inverse tangent in degrees.",
    description: "`atan2d` computes a real four-quadrant angle from broadcast-compatible Y and X coordinates, including compatible tables and timetables.",
    keywords: &["atan2d", "inverse tangent", "degrees", "quadrants", "broadcast"],
    related: &["atan2", "atand", "tand", "rad2deg"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink {
        label: "MATLAB reference",
        target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/double.atan2d.html"),
    }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Degree atan2 runtime",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/atan2d.rs"),
        }],
        verification: &[],
        notes: &["Source-only author packet; executable examples, resident restoration and tabular execution remain qualification obligations."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
