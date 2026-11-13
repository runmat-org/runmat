use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("fftshift"),
    slug: Some("fftshift"),
    summary: "Shift zero-frequency components to spectrum centers in MATLAB and RunMat.",
    description: "`fftshift(X)` circularly shifts FFT output so the zero-frequency component moves to the center of each shifted dimension. The default all-dimension behavior and optional dimension selection follow MATLAB semantics.",
    keywords: &[
        "fftshift",
        "fourier transform",
        "frequency centering",
        "spectrum",
        "gpu",
    ],
    related: &[
        "fft",
        "ifft",
        "ifftshift",
        "circshift",
        "gpuarray",
        "gather",
        "fft2",
        "ifft2",
    ],
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
                "When called without a dimension list, `fftshift` shifts along every dimension by `floor(size(X, dim) / 2)`.",
                "`fftshift(X, DIM)` shifts one documented scalar dimension. Numeric vectors and logical masks are separately gated RunMat extensions.",
                "Dimensions of length 0 or 1 are left unchanged.",
                "Inputs can be real or complex and may already reside on the GPU (`gpuArray`).",
            ],
        },
        BuiltinDocumentationSection {
            heading: "GPU execution",
            paragraphs: &[
                "Uses the owning provider's `circshift` hook only after a read-only positive preflight. A false preflight permits host fallback before provider effects; admitted failures are terminal and never replayed.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "GPU residency",
            paragraphs: &[
                "A valid native result remains on the owning device. Preflight-unavailable work gathers once and restores through the same owner only when it can physically preserve the class. You can call `gpuArray` explicitly when you want a residency boundary.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "GPU behavior",
            paragraphs: &[
                "RunMat uses the input owner's `circshift` hook when preflight admits the request; otherwise it shifts through the exact host path. Scalars and empty arrays remain on their existing device without a hook call.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1",
            title: "Centering the spectrum of a 1-D FFT result with even length",
            program: "x = [0 1 2 3 4 5 6 7];\nfx = fft(x);\ny = fftshift(fx)",
            display_output: Some(
                "y = [-4 -4-1.6569i -4-4i -4-9.6569i 28 -4+9.6569i -4+4i -4+1.6569i]",
            ),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(max(abs(ifftshift(y)-fx)) < 1e-10);",
            },
        },
        BuiltinExample {
            id: "example-2",
            title: "Handling odd-length vectors",
            program: "x = 1:5;\ny = fftshift(x)",
            display_output: Some("y = [4 5 1 2 3]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(y,[4 5 1 2 3]));",
            },
        },
        BuiltinExample {
            id: "example-3",
            title: "Centering both axes of a 2-D FFT",
            program: "A = [1 2 3; 4 5 6];\nC = fftshift(A)",
            display_output: Some("C =\n     6     4     5\n     3     1     2"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(C,[6 4 5;3 1 2]));",
            },
        },
        BuiltinExample {
            id: "example-4",
            title: "Shifting only one dimension of a matrix",
            program: "A = [1 2 3; 4 5 6];\nrowCentered = fftshift(A, 1)   % shift rows only",
            display_output: Some("rowCentered =\n     4     5     6\n     1     2     3"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(rowCentered,[4 5 6;1 2 3]));",
            },
        },
        BuiltinExample {
            id: "example-5",
            title: "Applying `fftshift` to a gpuArray spectrum",
            program: "G = gpuArray(0:7);\ncentered = fftshift(G);\nH = gather(centered)",
            display_output: Some("H = [4 5 6 7 0 1 2 3]"),
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Wgpu,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(H,[4 5 6 7 0 1 2 3]));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "When should I call `fftshift`?",
            answer: "Call `fftshift` whenever you need to center FFT outputs before visualising spectra, computing radial averages, or applying filters that expect zero frequency in the middle of the array.",
        },
        BuiltinDocumentationFaq {
            question: "Does `fftshift` modify the phase or magnitude of the FFT?",
            answer: "No. `fftshift` only reorders the samples. Magnitudes, phases, and the overall information content remain unchanged.",
        },
        BuiltinDocumentationFaq {
            question: "How do I undo `fftshift`?",
            answer: "Use `ifftshift`, which performs the inverse rearrangement. The sequence `ifftshift(fftshift(X))` returns `X` for all supported inputs.",
        },
        BuiltinDocumentationFaq {
            question: "Can I apply `fftshift` to only one dimension?",
            answer: "Pass a positive scalar dimension such as `fftshift(X,2)`. RunMat extension mode additionally accepts numeric dimension vectors and logical masks.",
        },
        BuiltinDocumentationFaq {
            question: "Does `fftshift` work with gpuArray inputs?",
            answer: "Yes. RunMat keeps data on the GPU whenever the provider exposes the `circshift` hook, matching MATLAB's `gpuArray` behaviour.",
        },
        BuiltinDocumentationFaq {
            question: "How does `fftshift` handle empty inputs?",
            answer: "Empty arrays are returned unchanged with identical shape metadata.",
        },
        BuiltinDocumentationFaq {
            question: "Can I use `fftshift` on logical arrays?",
            answer: "Yes. Logical arrays are shifted without changing their logical element type.",
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
            label: "ifftshift",
            target: BuiltinDocumentationLinkTarget::Builtin("ifftshift"),
        },
        BuiltinDocumentationLink {
            label: "circshift",
            target: BuiltinDocumentationLinkTarget::Builtin("circshift"),
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
            label: "`crates/runmat-runtime/src/builtins/math/fft/fftshift.rs`",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/fft/fftshift.rs",
            ),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Runtime FFT implementation",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/fft/fftshift.rs",
            ),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Runtime unit cases",
                location: "builtins::math::fft::fftshift::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Runtime integration case",
                location: "builtins::math::fft::fftshift::tests::fftshift_gpu_roundtrip",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: None,
};
