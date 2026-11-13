use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("im2double"),
    slug: Some("im2double"),
    summary: "Convert image samples to double precision.",
    description: "`im2double(I)` converts intensity and truecolor image samples to double. `im2double(I, \"indexed\")` applies indexed-image rules.",
    keywords: &["im2double", "image", "conversion", "indexed"],
    related: &["im2uint8", "im2uint16"],
    sections: &[
        BuiltinDocumentationSection { heading: "Classes and modes", paragraphs: &[
            "Normal mode maps uint8 and uint16 by their full class range and maps int16 from its signed range into [0,1]. Single and double values are not clamped; logical becomes 0 or 1.",
            "Indexed mode accepts double, logical, uint8, and uint16. Integer zero-based indices gain one in double; double and logical values remain unchanged.",
            "Normal mode admits double, single, int16, uint8, uint16, and logical images. Indexed mode admits double, uint8, uint16, and logical; unsupported integer and complex classes reject. Array geometry is preserved except that a one-element result is published as a scalar.",
        ]},
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Conversion uses authoritative host storage and is a fusion boundary. Resident inputs are gathered through their exact owner and supported results are restored to that owner when possible; explicit device intent is preserved or fails. The operation can suspend and throw.",
        ]},
    ],
    examples: &[
        BuiltinExample {
            id: "intensity", title: "Scale an integer image",
            program: "J = im2double(uint8([0 255]))",
            display_output: Some("J = [0 1]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(J,[0 1]));" },
        },
        BuiltinExample {
            id: "indexed", title: "Convert zero-based indices",
            program: "J = im2double(uint8([0 255]), \"indexed\")",
            display_output: Some("J = [1 256]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(J,[1 256]));" },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "im2uint8", target: BuiltinDocumentationLinkTarget::Builtin("im2uint8") },
        BuiltinDocumentationLink { label: "im2uint16", target: BuiltinDocumentationLinkTarget::Builtin("im2uint16") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/im2double.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Conversion implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/im2double.rs") },
            BuiltinDocumentationLink { label: "Shared color residency", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/common.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Image classes and indexed conversion", location: "crates/runmat-runtime/src/builtins/image/color/im2double/tests.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
