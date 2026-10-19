use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("gray2rgb"), slug: Some("gray2rgb"),
    summary: "Replicate a grayscale image into RGB planes.",
    description: "`gray2rgb(I)` is a RunMat-only helper that replicates an M-by-N grayscale image across three channels to form an M-by-N-by-3 RGB image.",
    keywords: &["gray2rgb", "grayscale", "rgb", "image"],
    related: &["rgb2gray"],
    sections: &[
        BuiltinDocumentationSection { heading: "Classes and shapes", paragraphs: &[
            "The source must be a two-dimensional real numeric or logical image. The output preserves its native class and exact integer values, including all eight integer classes; a logical input produces a logical RGB array. Empty matrices remain empty RGB images.",
            "The entire callable is a RunMat extension and requires extension mode in compatibility-sensitive programs.",
        ]},
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Resident inputs are gathered through their exact owner after shape and class admission. RGB plane replication returns a host result and is a fusion boundary. The call can suspend or throw.",
        ]},
    ],
    examples: &[
        BuiltinExample {
            id: "replicate", title: "Replicate grayscale channels",
            program: "I = uint8([0 255]); RGB = gray2rgb(I); size(RGB)",
            display_output: Some("ans = [1 2 3]"),
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(RGB),[1 2 3])); assert(isequal(RGB(:,:,1),I)); assert(isequal(RGB(:,:,2),I)); assert(isequal(RGB(:,:,3),I));" },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "rgb2gray", target: BuiltinDocumentationLinkTarget::Builtin("rgb2gray") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/gray2rgb.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Grayscale replication", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/gray2rgb.rs") },
            BuiltinDocumentationLink { label: "Color residency helper", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/common.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Logical, exact integer and shape replication", location: "crates/runmat-runtime/src/builtins/image/color/gray2rgb.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled color binding and fusion boundary", location: "crates/runmat-runtime/tests/image_color_catalog_binding.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
