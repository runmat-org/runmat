use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationLink, BuiltinDocumentationLinkTarget, BuiltinDocumentationSection,
    BuiltinDocumentationStatus, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Numerical behavior",
        paragraphs: &["Elementwise principal inverse sine in degrees. Real outputs follow [-90, 90]; complex input and out-of-real-domain values use the principal complex branch.", "Real single input retains single output storage, and double input retains double output storage. The result preserves scalar, array, empty and N-D shape. A real array promotes to complex storage if any element requires it."],
    },
    BuiltinDocumentationSection {
        heading: "Execution and compatibility",
        paragraphs: &["RunMat accepts exact-representable fixed-width integer, logical and character input only as explicit extensions. Resident input is computed after an owner-bound gather and restored through that same provider; an automatically resident real input needing complex promotion returns a host complex value. There is no dedicated degree-inverse provider kernel or fusion claim.", "Direct table, timetable, tall, sparse and symbolic overloads are not yet provided by this implementation."],
    },
];

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "scalar",
    title: "Compute inverse sine in degrees",
    program: "angle = asind(1)",
    display_output: Some("angle = 90"),
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: crate::BuiltinExampleFixture::None,
    requirements: crate::BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(abs(angle - 90) < 1e-12);",
    },
}];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("asind"),
    slug: Some("asind"),
    summary: "Compute inverse sine in degrees.",
    description: "Elementwise principal inverse sine in degrees. Real outputs follow [-90, 90]; complex input and out-of-real-domain values use the principal complex branch.",
    keywords: &["asind", "inverse trigonometry", "degrees", "elementwise"],
    related: &["asin", "sind", "rad2deg"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink {
        label: "MATLAB reference",
        target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/double.asind.html"),
    }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Inverse-degree runtime",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/inverse_degree/asind.rs"),
        }],
        verification: &[],
        notes: &["Source-only author packet; examples and provider routes require execution qualification."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
