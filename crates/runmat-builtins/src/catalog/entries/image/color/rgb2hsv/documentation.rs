use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("rgb2hsv"), slug: Some("rgb2hsv"),
    summary: "Convert RGB image or colormap values to HSV color space.",
    description: "`rgb2hsv(RGB)` converts M-by-N-by-3 truecolor RGB data or a double N-by-3 colormap into hue, saturation, and value channels.",
    keywords: &["rgb2hsv", "rgb", "hsv", "color", "image", "colormap"],
    related: &["hsv2rgb"],
    sections: &[
        BuiltinDocumentationSection { heading: "Classes and geometry", paragraphs: &[
            "Truecolor input must be a nonempty M-by-N-by-3 single, double, uint8, or uint16 array. A nonempty N-by-3 colormap is accepted only in double. Empty images, logical data, signed and wide unsigned integers, complex values, and other shapes reject.",
            "uint8 and uint16 samples are scaled by their full class range before conversion. Floating samples are clamped to [0,1]. Single truecolor output stays single; all other supported inputs produce double. HSV output retains the input image or colormap shape.",
        ]},
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Supported resident input is checked by owner, class, and shape before download. The typed gather, host conversion, and exact-owner restoration are one admitted whole-work route when its provider upload is supported; shared placement accounts for readback host workspace, physical retention, and completion-qualified feedback. Without that route, the existing host conversion and restoration behavior remains. The conversion is a fusion boundary; explicit device intent is preserved or fails. The call can suspend and throw.",
        ]},
    ],
    examples: &[
        BuiltinExample {
            id: "red", title: "Convert red RGB to HSV",
            program: "RGB = reshape([1 0 0],[1 1 3]); HSV = rgb2hsv(RGB); reshape(HSV,[1 3])",
            display_output: Some("ans = [0 1 1]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(HSV),[1 1 3])); assert(isequal(reshape(HSV,[1 3]),[0 1 1]));" },
        },
        BuiltinExample {
            id: "colormap", title: "Convert a double colormap",
            program: "HSV = rgb2hsv([1 0 0; 0 1 0]); size(HSV)",
            display_output: Some("ans = [2 3]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(HSV),[2 3])); assert(abs(HSV(2,1)-1/3)<1e-12);" },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "hsv2rgb", target: BuiltinDocumentationLinkTarget::Builtin("hsv2rgb") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/rgb2hsv.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "HSV conversion", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/rgb2hsv.rs") },
            BuiltinDocumentationLink { label: "RGB admission and residency", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/rgb_input.rs") },
            BuiltinDocumentationLink { label: "Resident whole-work presentation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/rgb_presentation.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Truecolor/colormap and resident precision tests", location: "crates/runmat-runtime/src/builtins/image/color/rgb2hsv.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled color binding and fusion boundary", location: "crates/runmat-runtime/tests/image_color_catalog_binding.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Resident owner-upload admission and feedback", location: "crates/runmat-runtime/src/builtins/image/color/rgb_presentation/tests.rs" },
        ],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
