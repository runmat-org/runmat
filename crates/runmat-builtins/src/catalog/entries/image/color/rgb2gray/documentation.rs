use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("rgb2gray"), slug: Some("rgb2gray"),
    summary: "Convert RGB image data to grayscale intensity using luminance weighting.",
    description: "`rgb2gray(RGB)` converts an M-by-N-by-3 truecolor image to grayscale using luminance weighting. A double N-by-3 colormap becomes three equal gray columns.",
    keywords: &["rgb2gray", "rgb", "grayscale", "luminance", "image"],
    related: &["im2double", "gray2rgb"],
    sections: &[
        BuiltinDocumentationSection { heading: "Classes and geometry", paragraphs: &[
            "Truecolor input must be a nonempty M-by-N-by-3 single, double, uint8, or uint16 array. A nonempty N-by-3 colormap is accepted only in double. Empty images, logical data, signed and wide unsigned integers, complex values, and other shapes reject.",
            "Red, green, and blue use weights 0.298936021293775, 0.587043074451121, and 0.114020904255103. Truecolor output has M-by-N shape; a double colormap yields N-by-3 equal grayscale columns. A one-pixel truecolor output is published as a scalar.",
            "Multi-pixel single/double truecolor output retains its input class; the one-pixel single result is published as a double scalar. uint8/uint16 output retains its integer class using nearest rounding with ties away from zero and saturation.",
        ]},
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Supported resident input is checked by owner, class, and shape before download. The typed gather, host conversion, and exact-owner restoration are one admitted whole-work route when its provider upload is supported; shared placement accounts for readback host workspace, physical retention, and completion-qualified feedback. Without that route, the existing host conversion and restoration behavior remains. The conversion is a fusion boundary; explicit device intent is preserved or fails. The call can suspend and throw.",
        ]},
    ],
    examples: &[
        BuiltinExample {
            id: "red-pixel", title: "Convert a red RGB pixel",
            program: "RGB = reshape(uint8([255 0 0]),[1 1 3]); I = rgb2gray(RGB)",
            display_output: Some("I = uint8(76)"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(I,uint8(76)));" },
        },
        BuiltinExample {
            id: "colormap", title: "Gray colormap keeps three columns",
            program: "G = rgb2gray([1 0 0; 0 1 0]); size(G)",
            display_output: Some("ans = [2 3]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(G),[2 3])); assert(max(abs(G(:,1)-G(:,2))) < 1e-12); assert(max(abs(G(:,2)-G(:,3))) < 1e-12);" },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "im2double", target: BuiltinDocumentationLinkTarget::Builtin("im2double") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/rgb2gray.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Luminance conversion", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/rgb2gray.rs") },
            BuiltinDocumentationLink { label: "RGB admission and residency", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/rgb_input.rs") },
            BuiltinDocumentationLink { label: "Resident whole-work presentation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/rgb_presentation.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Truecolor/colormap and resident class tests", location: "crates/runmat-runtime/src/builtins/image/color/rgb2gray.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled color binding and fusion boundary", location: "crates/runmat-runtime/tests/image_color_catalog_binding.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Resident owner-upload admission and feedback", location: "crates/runmat-runtime/src/builtins/image/color/rgb_presentation/tests.rs" },
        ],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
