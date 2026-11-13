use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("fftn"),
    slug: Some("fftn"),
    summary: "Compute an N-dimensional discrete Fourier transform.",
    description: "`fftn(X)` transforms every dimension of `X`. `fftn(X,SIZE)` pads or truncates each transformed dimension according to a positive integer size vector whose length is at least `ndims(X)`.",
    keywords: &["fftn", "n-dimensional fft", "fourier transform", "gpu"],
    related: &["fft", "fft2", "ifftn", "fftshift"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Provider and fusion contract",
            paragraphs: &[
                "The provider hook advertises f32 and f64 floating precision with MATLAB-style input shape semantics; integer and logical behavior follows the separate documented conversion or exact-reordering rules.",
                "This operation is a fusion boundary, not an elementwise or reduction kernel. No fused FFT kernel is advertised, and provider launch details remain internal.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "Integer inputs through 32 bits and logical inputs are documented and produce double output; single input produces single output.",
                "`SIZE` must be a row or column vector of positive integers with at least one entry per input dimension.",
                "RunMat extension mode can admit exact int64 and uint64 structural controls and positive SIZE vectors shorter than `ndims(X)` through independent gates.",
                "gpuArray results remain resident when a provider hook succeeds or when RunMat completes a supported fallback on the host.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "GPU execution",
            paragraphs: &[
                "Runs provider-backed fft_dim passes when available; host fallback restores the result to the provider that owns the input.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1",
            title: "Transforming every dimension of a volume",
            program: "X = rand(4, 5, 6);\nY = fftn(X)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(Y),size(X)));",
            },
        },
        BuiltinExample {
            id: "example-2",
            title: "Padding an N-dimensional transform",
            program: "X = rand(4, 5, 6);\nY = fftn(X, [8 8 8])",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(Y),[8 8 8]));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What shape does `fftn(X,SIZE)` return?",
            answer: "Its transformed dimensions have the lengths listed in `SIZE`; the compatibility form requires `length(SIZE) >= ndims(X)`.",
        },
        BuiltinDocumentationFaq {
            question: "What precision is used for integer input?",
            answer: "Documented integer classes enter the double-precision FFT domain and return double output.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "fft",
            target: BuiltinDocumentationLinkTarget::Builtin("fft"),
        },
        BuiltinDocumentationLink {
            label: "fft2",
            target: BuiltinDocumentationLinkTarget::Builtin("fft2"),
        },
        BuiltinDocumentationLink {
            label: "ifftn",
            target: BuiltinDocumentationLinkTarget::Builtin("ifftn"),
        },
        BuiltinDocumentationLink {
            label: "fftshift",
            target: BuiltinDocumentationLinkTarget::Builtin("fftshift"),
        },
        BuiltinDocumentationLink {
            label: "Open a ticket",
            target: BuiltinDocumentationLinkTarget::External(
                "https://github.com/runmat-org/runmat/issues/new/choose",
            ),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Runtime FFT implementation",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/fft/fftn.rs",
            ),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Runtime unit cases",
                location: "builtins::math::fft::fftn::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Runtime integration case",
                location: "builtins::math::fft::fftn::tests::fftn_matches_sequential_fft_on_3d",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: None,
};
