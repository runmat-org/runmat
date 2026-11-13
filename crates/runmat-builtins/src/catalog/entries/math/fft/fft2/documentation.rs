use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("fft2"),
    slug: Some("fft2"),
    summary: "Compute two-dimensional discrete Fourier transforms in MATLAB and RunMat.",
    description: "`fft2(X)` computes the two-dimensional discrete Fourier transform of `X`. It is equivalent to applying `fft` along the first two dimensions, with shape and ordering behavior consistent with MATLAB and RunMat.",
    keywords: &[
        "fft2",
        "2d fft",
        "two dimensional fourier transform",
        "image frequency analysis",
        "gpu",
    ],
    related: &[
        "fft",
        "ifft",
        "fftshift",
        "gpuarray",
        "gather",
        "ifft2",
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
                "`fft2(X)` transforms along the first and second dimensions that have size greater than one.",
                "`fft2(X, M, N)` zero-pads or truncates `X` to `M` rows and `N` columns before evaluating the 2-D transform.",
                "RunMat extension: `fft2(X, SIZE)` accepts a two-element vector (or scalar) specifying the transform lengths when RunMat extensions are enabled.",
                "Real inputs produce complex outputs; complex inputs are transformed element-wise with no additional conversion.",
                "Higher-dimensional inputs are transformed slice-by-slice across trailing dimensions, matching MATLAB behaviour.",
                "MATLAB-compatible `M` and `N` controls are positive integer scalars. Empty and zero transform sizes are separately gated RunMat extensions.",
                "GPU arrays execute on-device when the provider advertises the `fft_dim` hook; otherwise RunMat gathers the data and performs the transform on the host using `rustfft`.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "GPU execution",
            paragraphs: &[
                "Runs two provider-backed fft_dim passes when available; otherwise the runtime gathers to the host.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "GPU residency",
            paragraphs: &[
                "You usually do NOT need to call `gpuArray` manually. The fusion planner and native acceleration layer keep tensors on the GPU when a provider offers FFT kernels. If the provider lacks `fft_dim`, RunMat gathers inputs, evaluates the FFT pair on the host, and returns a MATLAB-compatible complex tensor. You can still use `gpuArray` for explicit residency control, particularly when interoperating with MATLAB code that expects it.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1",
            title: "Computing the 2-D FFT of a small matrix",
            program: "X = [1 2; 3 4];\nY = fft2(X)",
            display_output: Some("Y =\n    10 + 0i   -2 + 0i\n    -4 + 0i    0 + 0i"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(max(abs(Y(:)-[10;-4;-2;0])) < 1e-10);",
            },
        },
        BuiltinExample {
            id: "example-2",
            title: "Zero-padding an image patch before `fft2`",
            program: "patch = [1 0 1; 0 1 0; 1 0 1];\nF = fft2(patch, 8, 8)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(F),[8 8]));",
            },
        },
        BuiltinExample {
            id: "example-3",
            title: "Specifying transform lengths with a size vector",
            program: "X = rand(4, 6);\nF = fft2(X, [8 4]);   % requires RunMat extension mode",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(F),[8 4]));",
            },
        },
        BuiltinExample {
            id: "example-4",
            title: "Using `fft2` on `gpuArray` data",
            program: "G = gpuArray(rand(256, 256));\nF = fft2(G);\nR = gather(F)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Wgpu,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "expected = fft2(gather(G)); assert(isequal(size(R),[256 256])); assert(norm(R(:)-expected(:))/max(1,norm(expected(:))) < 1e-3);",
            },
        },
        BuiltinExample {
            id: "example-5",
            title: "Applying `fft2` to each slice of a 3-D volume",
            program: "V = rand(64, 64, 10);\nspectra = fft2(V)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(spectra),size(V)));",
            },
        },
        BuiltinExample {
            id: "example-6",
            title: "Verifying `fft2` against sequential `fft` calls",
            program: "X = rand(5, 7);\nsequential = fft(fft(X, [], 1), [], 2);\ndirect = fft2(X)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(max(abs(direct(:)-sequential(:))) < 1e-10);",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Is `fft2(X)` the same as `fft(fft(X, [], 1), [], 2)`?",
            answer: "Yes. RunMat literally performs the two sequential transforms so the results match MATLAB exactly.",
        },
        BuiltinDocumentationFaq {
            question: "How do zero-length transform sizes behave?",
            answer: "MATLAB-compatible mode requires positive `M` and `N`. RunMat extension mode can use `0` to produce a zero-sized complex tensor.",
        },
        BuiltinDocumentationFaq {
            question: "Can I use a single scalar for the size argument?",
            answer: "In RunMat extension mode, `fft2(X, K)` is shorthand for `fft2(X, [K K])`. The documented compatibility forms are `fft2(X)` and `fft2(X,M,N)`.",
        },
        BuiltinDocumentationFaq {
            question: "What happens when `X` has more than two dimensions?",
            answer: "RunMat applies `fft2` to every 2-D slice defined by the first two dimensions, leaving higher dimensions untouched.",
        },
        BuiltinDocumentationFaq {
            question: "Do I get complex outputs for real inputs?",
            answer: "Always. Even when the imaginary parts are zero, outputs are stored as complex tensors to mirror MATLAB semantics.",
        },
        BuiltinDocumentationFaq {
            question: "Will `fft2` run on the GPU automatically?",
            answer: "Yes if the input's owning provider implements `fft_dim` for its storage class. Otherwise RunMat gathers to the host, performs the transform with `rustfft`, and restores the result to that provider.",
        },
        BuiltinDocumentationFaq {
            question: "Does `fft2` normalise the output?",
            answer: "No. Like MATLAB, the forward FFT leaves scaling untouched; use `ifft2` for the inverse with `1/(M*N)` scaling.",
        },
        BuiltinDocumentationFaq {
            question: "Can I mix `[]` with explicit sizes (e.g., `fft2(X, [], 128)`)?",
            answer: "This is a RunMat extension. MATLAB-compatible mode requires positive scalar `M` and `N` values.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "fft",
            target: BuiltinDocumentationLinkTarget::Builtin("fft"),
        },
        BuiltinDocumentationLink {
            label: "ifft",
            target: BuiltinDocumentationLinkTarget::Builtin("ifft"),
        },
        BuiltinDocumentationLink {
            label: "fftshift",
            target: BuiltinDocumentationLinkTarget::Builtin("fftshift"),
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
            label: "ifft2",
            target: BuiltinDocumentationLinkTarget::Builtin("ifft2"),
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
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/fft/fft2.rs",
            ),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Runtime unit cases",
                location: "builtins::math::fft::fft2::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Runtime integration case",
                location: "builtins::math::fft::fft2::tests::fft2_gpu_roundtrip_matches_cpu",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: None,
};
