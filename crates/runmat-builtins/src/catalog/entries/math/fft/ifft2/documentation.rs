use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ifft2"),
    slug: Some("ifft2"),
    summary: "Compute two-dimensional inverse Fourier transforms in MATLAB and RunMat.",
    description: "`ifft2(X)` computes the two-dimensional inverse discrete Fourier transform of `X`. Documented integer data through 32 bits enters the double transform domain; host `int64` and `uint64` data and shorthand size forms are exactness-checked RunMat extensions.",
    keywords: &[
        "ifft2",
        "inverse 2d fft",
        "image reconstruction",
        "symmetric",
        "gpu",
    ],
    related: &[
        "fft2",
        "ifft",
        "fft",
        "gpuarray",
        "gather",
        "fftshift",
        "ifftshift",
    ],
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
                "`ifft2(X)` transforms along the first two dimensions whose sizes exceed one.",
                "`ifft2(X, M, N)` zero-pads or truncates the spectrum to `M` rows and `N` columns before the inverse.",
                "`ifft2(X, SIZE)` accepts a scalar or two-element vector describing the transform lengths.",
                "`ifft2(..., 'symmetric')` discards tiny imaginary parts and returns a real matrix when the spectrum is conjugate-symmetric. `'nonsymmetric'` keeps the complex result.",
                "Higher-dimensional inputs are processed slice-by-slice across trailing dimensions.",
                "Empty or zero size controls and scalar/two-element SIZE shorthand are independently gated RunMat extensions.",
                "Provider results must be fresh, correctly shaped, complex, physically precise, and owned by the input provider; only an unsupported hook falls back through exact host storage.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "GPU execution",
            paragraphs: &[
                "Uses the input handle's owning provider for two validated `ifft_dim` passes. Only an explicit unsupported-hook response selects exact host fallback; malformed, aliased, or wrong-owner results are terminal and safely released.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "GPU residency",
            paragraphs: &[
                "RunMat resolves execution through the provider that owns the input handle. Valid nonsymmetric floating results remain resident. Symmetric output and explicit unsupported-hook fallback may download internally; RunMat restores the result only when the owner can physically preserve its class, otherwise it returns the correctly typed host value.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1",
            title: "Reconstructing an image patch from its 2-D spectrum",
            program: "X = [1 2; 3 4];\nF = fft2(X);\nx = ifft2(F)",
            display_output: Some("x =\n    1.0000    2.0000\n    3.0000    4.0000"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(max(abs(x(:)-X(:))) < 1e-10);",
            },
        },
        BuiltinExample {
            id: "example-2",
            title: "Zero-padding before the inverse transform",
            program: "F = fft2([1 0; 0 1], 4, 4);\nspatial = ifft2(F, 4, 4)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(spatial),[4 4]));",
            },
        },
        BuiltinExample {
            id: "example-3",
            title: "Supplying transform lengths with a size vector",
            program: "Y = fft2(rand(3,4));\nX = ifft2(Y, [5 2]);   % pad rows to 5, truncate columns to 2",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(X),[5 2]));",
            },
        },
        BuiltinExample {
            id: "example-4",
            title: "Forcing a real-valued result with `'symmetric'`",
            program: "F = fft2([1 2; 3 4]);\nrealImage = ifft2(F, 'symmetric')",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(max(abs(realImage(:)-[1;3;2;4])) < 1e-10);",
            },
        },
        BuiltinExample {
            id: "example-5",
            title: "Running `ifft2` on `gpuArray` inputs",
            program: "G = gpuArray(fft2(peaks(64)));\nspatial = ifft2(G);\nresult = gather(spatial)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Wgpu,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(result),[64 64]));",
            },
        },
        BuiltinExample {
            id: "example-6",
            title: "Recovering each slice of a volume",
            program: "spectra = fft2(rand(16, 16, 10));\nvolume = ifft2(spectra)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(volume),[16 16 10]));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Is `ifft2(X)` equivalent to `ifft(ifft(X, [], 1), [], 2)`?",
            answer: "Yes. RunMat literally performs two sequential 1-D inverse transforms so the behaviour matches MATLAB exactly.",
        },
        BuiltinDocumentationFaq {
            question: "How do zero-length sizes behave?",
            answer: "Passing `0` for either transform length produces an output with zero elements along that dimension, just like MATLAB.",
        },
        BuiltinDocumentationFaq {
            question: "Can I mix `[]` with explicit sizes (e.g., `ifft2(X, [], 64)`)?",
            answer: "Yes. `[]` leaves that dimension unchanged while the other argument controls padding or truncation.",
        },
        BuiltinDocumentationFaq {
            question: "What does the `'symmetric'` flag do?",
            answer: "It tells RunMat to coerce the result to real values, assuming the spectrum is conjugate-symmetric. Imaginary parts are dropped.",
        },
        BuiltinDocumentationFaq {
            question: "What happens when the input is real-valued?",
            answer: "RunMat promotes the data to complex with zero imaginary parts before applying the inverse. The output can still be coerced to real with `'symmetric'`.",
        },
        BuiltinDocumentationFaq {
            question: "Will `ifft2` run on the GPU automatically?",
            answer: "Yes when the input handle's owning provider exposes `ifft_dim` and returns valid fresh results. An explicit unsupported-hook response falls back through an exact owner download; provider failures or malformed results are not silently gathered away.",
        },
        BuiltinDocumentationFaq {
            question: "Does `ifft2` normalise the result?",
            answer: "Yes. The builtin divides by the product of the transform lengths so that `ifft2(fft2(X))` reproduces `X`.",
        },
        BuiltinDocumentationFaq {
            question: "Can I pass `gpuArray` size vectors or symmetry flags?",
            answer: "No. Length arguments must be host scalars or vectors, and the symmetry flag must be a host string, mirroring MATLAB’s restrictions.",
        },
        BuiltinDocumentationFaq {
            question: "How are higher-dimensional arrays handled?",
            answer: "Transformations are applied to every 2-D slice defined by the first two dimensions; trailing dimensions are preserved unchanged.",
        },
        BuiltinDocumentationFaq {
            question: "Does `'nonsymmetric'` change the result?",
            answer: "It simply states the default behaviour (return complex outputs) but is accepted for MATLAB compatibility.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "fft2",
            target: BuiltinDocumentationLinkTarget::Builtin("fft2"),
        },
        BuiltinDocumentationLink {
            label: "ifft",
            target: BuiltinDocumentationLinkTarget::Builtin("ifft"),
        },
        BuiltinDocumentationLink {
            label: "fft",
            target: BuiltinDocumentationLinkTarget::Builtin("fft"),
        },
        BuiltinDocumentationLink {
            label: "gpuArray",
            target: BuiltinDocumentationLinkTarget::Builtin("gpuarray"),
        },
        BuiltinDocumentationLink {
            label: "gather",
            target: BuiltinDocumentationLinkTarget::Builtin("gather"),
        },
        BuiltinDocumentationLink {
            label: "fftshift",
            target: BuiltinDocumentationLinkTarget::Builtin("fftshift"),
        },
        BuiltinDocumentationLink {
            label: "ifftshift",
            target: BuiltinDocumentationLinkTarget::Builtin("ifftshift"),
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
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/fft/ifft2.rs",
            ),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Runtime unit cases",
                location: "builtins::math::fft::ifft2::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Runtime integration case",
                location: "builtins::math::fft::ifft2::tests::ifft2_gpu_roundtrip_matches_cpu",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: None,
};
