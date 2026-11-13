use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("hsv2rgb"), slug: Some("hsv2rgb"),
    summary: "Convert HSV image or colormap values to RGB color space.",
    description: "`hsv2rgb(HSV)` converts M-by-N-by-3 HSV images or N-by-3 HSV colormaps to RGB values.",
    keywords: &["hsv2rgb", "hsv", "rgb", "color", "image", "colormap"],
    related: &["rgb2hsv"],
    sections: &[
        BuiltinDocumentationSection { heading: "Classes and geometry", paragraphs: &[
            "Nonempty M-by-N-by-3 images accept single, double, or logical values. Nonempty N-by-3 colormaps accept single or double. Typed integers, logical colormaps, complex data, empty images, and other geometry reject.",
            "Hue wraps modulo one; saturation and value clamp to [0,1]. Single input produces single RGB; double and logical input produce double. Shape is preserved.",
        ]},
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "This is a host conversion and fusion boundary. Admitted numeric resident input is checked by owner, class, and shape before preparation and downloaded once, then the result is restored through the original owner when possible. Explicit device intent is preserved or fails. Logical resident data uses the owner-aware gather path.",
        ]},
    ],
    examples: &[
        BuiltinExample {
            id: "red", title: "Convert red HSV to RGB",
            program: "HSV = reshape([0 1 1],[1 1 3]); RGB = hsv2rgb(HSV); reshape(RGB,[1 3])",
            display_output: Some("ans = [1 0 0]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(RGB),[1 1 3])); assert(isequal(reshape(RGB,[1 3]),[1 0 0]));" },
        },
        BuiltinExample {
            id: "colormap", title: "Wrap hue in an HSV colormap",
            program: "RGB = hsv2rgb([1 1 1; 1/3 1 1]); size(RGB)",
            display_output: Some("ans = [2 3]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(RGB),[2 3])); assert(abs(RGB(1,1)-1)<1e-12); assert(abs(RGB(2,2)-1)<1e-12);" },
        },
    ],
    example_exemption: None, faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "rgb2hsv", target: BuiltinDocumentationLinkTarget::Builtin("rgb2hsv") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/hsv2rgb.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "HSV conversion", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/hsv2rgb.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "HSV class, colormap, resident and boundary tests", location: "crates/runmat-runtime/src/builtins/image/color/hsv2rgb.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled color binding and fusion boundary", location: "crates/runmat-runtime/tests/image_color_catalog_binding.rs" },
        ],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
