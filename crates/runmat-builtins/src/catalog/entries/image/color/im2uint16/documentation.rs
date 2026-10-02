use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("im2uint16"),
    slug: Some("im2uint16"),
    summary: "Convert image samples to uint16.",
    description: "`im2uint16(I)` converts intensity and truecolor image samples to uint16. `im2uint16(I, \"indexed\")` applies indexed-image rules.",
    keywords: &["im2uint16", "image", "conversion", "indexed"],
    related: &["im2double", "im2uint8"],
    sections: &[
        BuiltinDocumentationSection { heading: "Classes and modes", paragraphs: &[
            "Normal mode preserves uint16; expands uint8 to its full 16-bit range; offsets int16 by 32768; clamps floating values to [0,1], scales by 65535, and rounds; logical false/true becomes 0/65535.",
            "Indexed mode preserves uint8 and uint16 indices and subtracts one from one-based double indices. Logical values become 0 or 1.",
            "Normal mode admits double, single, int16, uint8, uint16, and logical images. Indexed mode admits double, uint8, uint16, and logical; unsupported integer and complex classes reject. Array geometry is preserved except that a one-element result is published as a scalar.",
        ]},
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Conversion uses authoritative host storage and is a fusion boundary. Resident inputs are gathered through their exact owner and supported results are restored to that owner when possible; explicit device intent is preserved or fails. The operation can suspend and throw.",
        ]},
    ],
    examples: &[
        BuiltinExample {
            id: "intensity", title: "Expand an eight-bit image",
            program: "J = im2uint16(uint8([0 255]))",
            display_output: Some("J = uint16([0 65535])"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(J,uint16([0 65535])));" },
        },
        BuiltinExample {
            id: "indexed", title: "Convert one-based indices",
            program: "J = im2uint16([1 65536], \"indexed\")",
            display_output: Some("J = uint16([0 65535])"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(J,uint16([0 65535])));" },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "im2double", target: BuiltinDocumentationLinkTarget::Builtin("im2double") },
        BuiltinDocumentationLink { label: "im2uint8", target: BuiltinDocumentationLinkTarget::Builtin("im2uint8") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/im2uint16.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Conversion implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/im2uint16.rs") },
            BuiltinDocumentationLink { label: "Shared color residency", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/common.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Image classes and indexed conversion", location: "crates/runmat-runtime/src/builtins/image/color/im2uint16/tests.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
