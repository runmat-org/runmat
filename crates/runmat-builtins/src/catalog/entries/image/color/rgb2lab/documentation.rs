use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("rgb2lab"), slug: Some("rgb2lab"),
    summary: "Convert RGB image, image-stack, or colormap values to CIE L*a*b*.",
    description: "`rgb2lab(RGB)` converts RGB values to CIE L*a*b*. ColorSpace and WhitePoint name-value pairs select source primaries and reference white.",
    keywords: &["rgb2lab", "rgb", "lab", "cielab", "color", "image"],
    related: &["lab2rgb"],
    sections: &[
        BuiltinDocumentationSection { heading: "Classes and geometry", paragraphs: &[
            "RGB accepts nonempty M-by-N-by-3 truecolor images, M-by-N-by-3-by-P stacks, and nonempty K-by-3 colormaps in single, double, uint8, or uint16. An image stack may have an empty trailing stack axis while M and N remain positive. Logical, complex, signed and wide unsigned integers, and other geometry reject.",
            "uint8 and uint16 samples normalize by their full native range; floating samples clamp to [0,1]. Single input produces single Lab; double and integer input produce double. Shape is preserved, including stack axes.",
        ]},
        BuiltinDocumentationSection { heading: "Color controls", paragraphs: &[
            "ColorSpace accepts case-insensitive srgb (default), adobe-rgb-1998, prophoto-rgb, or linear-rgb. WhitePoint accepts A, C, E, D50, D55, D65 (default), ICC, or a finite positive single/double 1-by-3 XYZ row. Up to two name-value pairs are accepted; repeated options use the last value.",
        ]},
        BuiltinDocumentationSection { heading: "Execution and compatibility", paragraphs: &[
            "The channel-coupled host conversion is a fusion boundary. Parsed controls and admitted class, shape, and owner precede operand preparation. Resident output restores through the original owner when possible. Explicit gpuArray input requires the RunMat-only Rgb2labExplicitGpuExtension; automatic resident input may gather normally.",
        ]},
    ],
    examples: &[
        BuiltinExample {
            id: "white", title: "Convert RGB white to Lab",
            program: "RGB = reshape([1 1 1],[1 1 3]); LAB = rgb2lab(RGB); size(LAB)",
            display_output: Some("ans = [1 1 3]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(LAB),[1 1 3])); assert(abs(LAB(1,1,1)-100)<0.01); assert(abs(LAB(1,1,2))<0.01);" },
        },
        BuiltinExample {
            id: "controls", title: "Select a reference white",
            program: "LAB = rgb2lab([1 0 0],'ColorSpace','srgb','WhitePoint','D50'); size(LAB)",
            display_output: Some("ans = [1 3]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(LAB),[1 3])); assert(isfinite(LAB(1,1))); assert(LAB(1,1)>0);" },
        },
    ],
    example_exemption: None, faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "lab2rgb", target: BuiltinDocumentationLinkTarget::Builtin("lab2rgb") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/rgb2lab.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "RGB to Lab options and conversion", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/rgb2lab.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "RGB color-space, white-point, stack and owner tests", location: "crates/runmat-runtime/src/builtins/image/color/rgb2lab/tests.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled binding and fusion boundary", location: "crates/runmat-runtime/tests/image_color_catalog_binding.rs" },
        ],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
