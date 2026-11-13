use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ind2rgb"), slug: Some("ind2rgb"),
    summary: "Convert an indexed image and colormap to double RGB values.",
    description: "`ind2rgb(X, map)` looks up a nonempty double N-by-3 colormap for each indexed-image sample and appends a three-channel RGB dimension.",
    keywords: &["ind2rgb", "indexed image", "colormap", "rgb", "image"],
    related: &["rgb2hsv"],
    sections: &[
        BuiltinDocumentationSection { heading: "Classes, indices, and geometry", paragraphs: &[
            "X is an M-by-N single, double, uint8, or uint16 matrix, including an empty matrix. The colormap must be a nonempty double K-by-3 matrix. Logical, unsupported integer, complex, and nonmatrix inputs reject.",
            "Single and double indices are finite integers interpreted one-based. uint8 and uint16 indices are zero-based. Indices outside the colormap range clip to its first or last row. Floating NaN, infinity, and fractional indices raise InvalidIndex.",
            "The result is an M-by-N-by-3 double array. A one-pixel input still has three RGB samples; no scalar publication occurs.",
        ]},
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "The host lookup is a fusion boundary. Class, shape, owner, and known host index validity are admitted before operand preparation. Resident operands are gathered through their owning providers; the double result is restored through the original selected owner when possible, preserving explicit device intent or failing.",
        ]},
    ],
    examples: &[
        BuiltinExample {
            id: "indexed", title: "Convert a small indexed image",
            program: "X = [1 2]; map = [1 0 0; 0 1 0]; RGB = ind2rgb(X,map); size(RGB)",
            display_output: Some("ans = [1 2 3]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(RGB),[1 2 3])); assert(RGB(1,1,1)==1); assert(RGB(1,2,2)==1);" },
        },
        BuiltinExample {
            id: "uint8", title: "Use zero-based uint8 indices",
            program: "X = uint8([0 1]); map = [1 0 0; 0 1 0]; RGB = ind2rgb(X,map); reshape(RGB(1,:,2),[1 2])",
            display_output: Some("ans = [0 1]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(reshape(RGB(1,:,2),[1 2]),[0 1]));" },
        },
    ],
    example_exemption: None, faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "rgb2hsv", target: BuiltinDocumentationLinkTarget::Builtin("rgb2hsv") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/ind2rgb.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Indexed lookup and admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/image/color/ind2rgb.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Indexed classes, clipping, invalid indices and exact owner", location: "crates/runmat-runtime/src/builtins/image/color/ind2rgb.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled binding and fusion boundary", location: "crates/runmat-runtime/tests/image_color_catalog_binding.rs" },
        ],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
