use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("lab2rgb"), slug: Some("lab2rgb"),
    summary: "Convert CIE L*a*b* image or colormap values to sRGB under D65.",
    description: "`lab2rgb(LAB)` converts single or double M-by-N-by-3 Lab images or N-by-3 Lab colormaps to sRGB. `OutputType` selects the output class.",
    keywords: &["lab2rgb", "lab", "cielab", "rgb", "color", "image"],
    related: &["rgb2lab"],
    sections: &[
        BuiltinDocumentationSection { heading: "Classes, options, and geometry", paragraphs: &[
            "Input must be a nonempty M-by-N-by-3 image or N-by-3 colormap in single or double. Logical, integer, complex, empty, and stacked inputs reject.",
            "Without OutputType, single remains single and double remains double. OutputType accepts case-insensitive double, single, uint8, or uint16. Floating output preserves out-of-gamut values; integer output clips normalized RGB to [0,1], rounds to the native class range, and publishes exact integer storage.",
        ]},
        BuiltinDocumentationSection { heading: "Execution and compatibility", paragraphs: &[
            "Conversion runs on the host as a fusion boundary. Admitted resident input is checked by owner, class, and shape before preparation and restored through the original owner when possible. Explicit gpuArray input requires the RunMat-only Lab2rgbExplicitGpuExtension; automatic resident input may gather normally.",
        ]},
    ],
    examples: &[
        BuiltinExample {
            id: "white", title: "Convert neutral Lab white",
            program: "LAB = reshape([100 0 0],[1 1 3]); RGB = lab2rgb(LAB); size(RGB)",
            display_output: Some("ans = [1 1 3]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(RGB),[1 1 3])); assert(abs(RGB(1,1,1)-1)<0.001);" },
        },
        BuiltinExample {
            id: "uint8", title: "Select native uint8 output",
            program: "LAB = reshape([0 0 0],[1 1 3]); RGB = lab2rgb(LAB,'OutputType','uint8'); size(RGB)",
            display_output: Some("ans = [1 1 3]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isa(RGB,'uint8')); assert(all(reshape(RGB,[1 3])==0));" },
        },
    ],
    example_exemption: None, faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "rgb2lab", target: BuiltinDocumentationLinkTarget::Builtin("rgb2lab") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/lab2rgb.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Lab conversion and typed OutputType", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/lab2rgb.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Lab precision, OutputType, owner and option errors", location: "crates/runmat-runtime/src/builtins/image/color/lab2rgb/tests.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled binding and fusion boundary", location: "crates/runmat-runtime/tests/image_color_catalog_binding.rs" },
        ],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
