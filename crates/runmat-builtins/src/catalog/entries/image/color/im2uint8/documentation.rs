use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("im2uint8"),
    slug: Some("im2uint8"),
    summary: "Convert image samples to uint8.",
    description: "`im2uint8(I)` converts intensity and truecolor image samples to uint8. `im2uint8(I, \"indexed\")` applies indexed-image rules.",
    keywords: &["im2uint8", "image", "conversion", "indexed"],
    related: &["im2double", "im2uint16"],
    sections: &[
        BuiltinDocumentationSection { heading: "Classes and modes", paragraphs: &[
            "Normal mode preserves uint8; rescales uint16 and int16 from their full ranges; clamps floating values to [0,1], scales by 255, and rounds; logical false/true becomes 0/255.",
            "Indexed mode preserves uint8 indices, converts uint16 indices with a range check, and subtracts one from one-based double indices. Logical values become 0 or 1.",
            "Normal mode admits double, single, int16, uint8, uint16, and logical images. Indexed mode admits double, uint8, uint16, and logical; unsupported integer and complex classes reject. Array geometry is preserved except that a one-element result is published as a scalar.",
        ]},
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Conversion uses authoritative host storage and is a fusion boundary. Resident inputs are gathered through their exact owner and supported results are restored to that owner when possible; explicit device intent is preserved or fails. The operation can suspend and throw.",
        ]},
    ],
    examples: &[
        BuiltinExample {
            id: "intensity", title: "Quantize a normalized image",
            program: "J = im2uint8([0 1])",
            display_output: Some("J = uint8([0 255])"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(J,uint8([0 255])));" },
        },
        BuiltinExample {
            id: "indexed", title: "Convert one-based indices",
            program: "J = im2uint8([1 256], \"indexed\")",
            display_output: Some("J = uint8([0 255])"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(J,uint8([0 255])));" },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "im2double", target: BuiltinDocumentationLinkTarget::Builtin("im2double") },
        BuiltinDocumentationLink { label: "im2uint16", target: BuiltinDocumentationLinkTarget::Builtin("im2uint16") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/im2uint8.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Conversion implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/im2uint8.rs") },
            BuiltinDocumentationLink { label: "Shared color residency", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/common.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Image classes and indexed conversion", location: "crates/runmat-runtime/src/builtins/image/color/im2uint8/tests.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
