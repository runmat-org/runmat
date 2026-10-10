use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ifftn"),
    slug: Some("ifftn"),
    summary: "Compute N-dimensional inverse Fourier transforms in MATLAB and RunMat.",
    description: "`ifftn(X)` computes the N-dimensional inverse discrete Fourier transform. Documented integer data through 32 bits enters the double transform domain; host `int64` and `uint64` data and wide size controls are exactness-checked RunMat extensions.",
    keywords: &[
        "ifftn",
        "inverse n-dimensional fft",
        "multidimensional",
        "symmetric",
        "gpu",
    ],
    related: &["fftn", "ifft", "ifft2", "ifftshift"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Provider and fusion contract",
            paragraphs: &[
                "The provider hook advertises f32 and f64 floating precision with no broadcasting; integer and logical behavior follows the separate documented conversion or exact-reordering rules.",
                "This operation is a fusion boundary, not an elementwise or reduction kernel. No fused FFT kernel is advertised, and provider launch details remain internal.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`ifftn(X)` transforms every dimension.",
                "`ifftn(X, SIZE)` pads or truncates each dimension to the requested positive integer size.",
                "`ifftn(..., 'symmetric')` returns real output; `'nonsymmetric'` preserves the default complex result.",
                "Documented integer and logical data produces double numeric class while preserving N-D shape.",
                "Provider outputs are accepted only with the expected owner, device, shape, complex storage, precision, and nonaliasing lifetime.",
                "Symmetric output and unsupported-hook fallback download through the input owner and restore residency only when the provider can physically preserve the result class.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "GPU execution",
            paragraphs: &[
                "Runs sequential validated `ifft_dim` passes through the input handle's owner. Only explicit unsupported-hook responses select exact host fallback.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "GPU residency",
            paragraphs: &[
                "RunMat resolves every transform pass through the input owner. Intermediate handles are released after each validated pass without touching the source handle; provider failures, malformed results, and aliases are terminal rather than silently gathered away.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1",
            title: "Invert a three-dimensional transform",
            program: "F = fftn(rand(4, 5, 6));\nX = ifftn(F);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(X),[4 5 6]));",
            },
        },
        BuiltinExample {
            id: "example-2",
            title: "Use integer input",
            program: "X = ifftn(uint16(reshape(1:8, [2 2 2])));",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(X),[2 2 2]));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink {
            label: "fftn",
            target: BuiltinDocumentationLinkTarget::Builtin("fftn"),
        },
        BuiltinDocumentationLink {
            label: "ifft",
            target: BuiltinDocumentationLinkTarget::Builtin("ifft"),
        },
        BuiltinDocumentationLink {
            label: "ifft2",
            target: BuiltinDocumentationLinkTarget::Builtin("ifft2"),
        },
        BuiltinDocumentationLink {
            label: "ifftshift",
            target: BuiltinDocumentationLinkTarget::Builtin("ifftshift"),
        },
        BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/math/fft/ifftn.rs`",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/fft/ifftn.rs",
            ),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Runtime FFT implementation",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/fft/ifftn.rs",
            ),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Runtime unit cases",
                location: "builtins::math::fft::ifftn::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Runtime integration case",
                location: "crates/runmat-vm/tests/integer_multidimensional_fft_and_table_selection.rs",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: None,
};
