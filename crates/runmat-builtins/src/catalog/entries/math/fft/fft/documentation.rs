use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("fft"),
    slug: Some("fft"),
    summary: "Compute discrete Fourier transforms in MATLAB and RunMat.",
    description: "`fft(X)` computes the discrete Fourier transform of input data. For arrays, it operates along the first non-singleton dimension by default, with explicit length/dimension forms following MATLAB semantics.",
    keywords: &["fft", "fourier transform", "complex", "zero padding", "gpu"],
    related: &[
        "ifft",
        "fftshift",
        "abs",
        "angle",
        "gpuarray",
        "gather",
        "fft2",
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
                "`fft(X)` transforms along the first dimension whose size is greater than 1.",
                "`fft(X, n)` zero-pads or truncates `X` to length `n` before transforming along the default dimension.",
                "`fft(X, n, dim)` applies the transform along dimension `dim`.",
                "Integer inputs through 32 bits are transformed in double precision; single inputs produce single output and all other documented numeric classes produce double output.",
                "Without a length override, empty inputs remain empty; zero-padding with `n` pads the input with zeros before the transform and does not generally produce zero-valued spectra.",
                "GPU arrays are gathered to the host when the selected provider has no FFT implementation.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "GPU execution",
            paragraphs: &[
                "Falls back to host execution when the active acceleration provider does not expose an FFT hook.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1",
            title: "Computing the FFT of a real time-domain vector",
            program: "x = [1 2 3 4];\nY = fft(x)",
            display_output: Some(
                "Y =\n  Columns 1 through 4\n   10 + 0i  -2 + 2i  -2 + 0i  -2 - 2i",
            ),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(max(abs(Y - [10 -2+2i -2 -2-2i])) < 1e-10);",
            },
        },
        BuiltinExample {
            id: "example-2",
            title: "Applying fft column-wise to a matrix",
            program: "A = [1 2 3; 4 5 6];\nF = fft(A)",
            display_output: Some("F =\n   5 + 0i   7 + 0i   9 + 0i\n  -3 + 0i  -3 + 0i  -3 + 0i"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(F),[2 3])); assert(max(abs(F(1,:) - [5 7 9])) < 1e-10);",
            },
        },
        BuiltinExample {
            id: "example-3",
            title: "Zero-padding before the FFT",
            program: "x = [1 2 3];\nY = fft(x, 5)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(Y),[1 5])); assert(abs(Y(1)-6)<1e-10);",
            },
        },
        BuiltinExample {
            id: "example-4",
            title: "Selecting the transform dimension for a row vector",
            program: "x = [1 2 3 4];\nY = fft(x, [], 2)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(Y),size(x))); assert(abs(Y(1)-10)<1e-10);",
            },
        },
        BuiltinExample {
            id: "example-5",
            title: "FFT of a complex-valued signal",
            program: "t = 0:3;\nx = exp(1i * pi/2 * t);\nY = fft(x)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(Y),size(x))); assert(abs(Y(2)-4)<1e-10);",
            },
        },
        BuiltinExample {
            id: "example-6",
            title: "FFT with gpuArray inputs",
            program: "g = gpuArray(rand(1, 1024));  % Residency is on the GPU\nG = fft(g);                   % Falls back to host if provider FFT hooks are unavailable\nresult = gather(G)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Wgpu,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(G),size(g))); assert(max(abs(gather(G)-fft(gather(g)))) < 1e-3);",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does `fft` always return complex values?",
            answer: "FFT results represent a complex spectrum. For gpuArray input the result is always stored as complex even when every imaginary component is zero.",
        },
        BuiltinDocumentationFaq {
            question: "What happens if I pass `[]` as the second argument?",
            answer: "Passing `[]` leaves the transform length unchanged. This is equivalent to omitting the `n` parameter.",
        },
        BuiltinDocumentationFaq {
            question: "Can I transform along a dimension larger than the current rank?",
            answer: "Yes. RunMat automatically treats trailing dimensions as length-1 and will create the requested dimension on output.",
        },
        BuiltinDocumentationFaq {
            question: "How does zero-padding work?",
            answer: "When `n` is larger than the size of `X` along the transform dimension, RunMat pads with zeros before evaluating the FFT.",
        },
        BuiltinDocumentationFaq {
            question: "What precision is used for the FFT?",
            answer: "RunMat computes FFTs in double precision on the host. Providers may use single or double precision depending on device capabilities.",
        },
        BuiltinDocumentationFaq {
            question: "Will RunMat run the FFT on my GPU automatically?",
            answer: "When a provider installs an FFT hook, RunMat executes on the GPU. Otherwise, the runtime gathers the data and performs the transform on the CPU.",
        },
        BuiltinDocumentationFaq {
            question: "Is inverse FFT (`ifft`) available?",
            answer: "Yes. Use `ifft` for the inverse transform.",
        },
        BuiltinDocumentationFaq {
            question: "How do I compute multi-dimensional FFTs?",
            answer: "Use `fft2` for the first two dimensions or `fftn` for an N-dimensional transform.",
        },
        BuiltinDocumentationFaq {
            question: "Does `fft` support complex strides or non-unit sampling intervals?",
            answer: "`fft` assumes unit spacing. You can multiply the result by appropriate phase factors to account for custom sampling intervals.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "ifft",
            target: BuiltinDocumentationLinkTarget::Builtin("ifft"),
        },
        BuiltinDocumentationLink {
            label: "fftshift",
            target: BuiltinDocumentationLinkTarget::Builtin("fftshift"),
        },
        BuiltinDocumentationLink {
            label: "abs",
            target: BuiltinDocumentationLinkTarget::Builtin("abs"),
        },
        BuiltinDocumentationLink {
            label: "angle",
            target: BuiltinDocumentationLinkTarget::Builtin("angle"),
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
            label: "fft2",
            target: BuiltinDocumentationLinkTarget::Builtin("fft2"),
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
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/fft/forward.rs",
            ),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Runtime unit cases",
                location: "builtins::math::fft::fft::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Runtime integration case",
                location: "builtins::math::fft::fft::tests::fft_gpu_roundtrip_matches_cpu",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: None,
};
