use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ifftshift"),
    slug: Some("ifftshift"),
    summary: "Shift centered spectra back to origin ordering in MATLAB and RunMat.",
    description: "`ifftshift(X)` circularly shifts data so zero-frequency components move back to origin ordering. Default and dimension-specific shifting behavior follow MATLAB semantics.",
    keywords: &[
        "ifftshift",
        "inverse fft shift",
        "frequency alignment",
        "gpu",
    ],
    related: &[
        "fftshift",
        "fft",
        "ifft",
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
                "When no dimensions are specified, every axis is shifted by `ceil(size(X, dim) / 2)`.",
                "A scalar dimension restricts shifting to one axis. RunMat additionally accepts a dimension vector or logical mask as a compatibility-gated extension.",
                "Odd-length dimensions shift by one additional element compared with `fftshift`, matching MATLAB's `ifftshift` parity rules.",
                "Works for real, complex, logical, and GPU-resident tensors.",
                "Empty arrays and scalars are returned unchanged.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "GPU execution",
            paragraphs: &[
                "A read-only provider preflight decides whether the owning provider can admit this `circshift` request. A false preflight permits exact host fallback before provider effects; a failure after admission is terminal and never replayed.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "GPU residency",
            paragraphs: &[
                "After positive preflight, `ifftshift` accepts only a fresh result with matching shape, class, storage, precision, device, and ownership. Preflight-unavailable work uses one exact internal download and class-preserving restoration; admitted provider failures and malformed results are terminal.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "GPU behavior",
            paragraphs: &[
                "RunMat uses the owning provider's `circshift` hook when preflight admits the request. Otherwise it performs the reorder on the host and restores residency only when the owner can physically preserve the class. No-op scalars and empty arrays remain on their existing device without a hook call.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1",
            title: "Undoing fftshift on an even-length spectrum",
            program: "x = 0:7;\ny = ifftshift(x)",
            display_output: Some("y = [4 5 6 7 0 1 2 3]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(y,[4 5 6 7 0 1 2 3]));",
            },
        },
        BuiltinExample {
            id: "example-2",
            title: "Undoing fftshift on an odd-length spectrum",
            program: "x = 1:5;\ny = ifftshift(x)",
            display_output: Some("y = [3 4 5 1 2]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(y,[3 4 5 1 2]));",
            },
        },
        BuiltinExample {
            id: "example-3",
            title: "Preparing data for inverse FFT",
            program: "fx = fft(rand(1, 8));\ncentered = fftshift(fx);\nrestored = ifftshift(centered);\nsignal = ifft(restored)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(max(abs(restored-fx)) < 1e-10);",
            },
        },
        BuiltinExample {
            id: "example-4",
            title: "Shifting only selected dimensions",
            program: "A = reshape(1:12, 3, 4);\nrowsShifted = ifftshift(A, 1);   % shift rows only\ncolsShifted = ifftshift(A, 2);   % shift columns only",
            display_output: Some(
                "rowsShifted =\n     2     5     8    11\n     3     6     9    12\n     1     4     7    10\n\ncolsShifted =\n     7    10     1     4\n     8    11     2     5\n     9    12     3     6",
            ),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(rowsShifted),size(A))); assert(isequal(size(colsShifted),size(A)));",
            },
        },
        BuiltinExample {
            id: "example-5",
            title: "Using ifftshift with gpuArray data",
            program: "G = gpuArray(0:7);\nshifted = ifftshift(G);\nhost = gather(shifted)",
            display_output: Some("host = [4 5 6 7 0 1 2 3]"),
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Wgpu,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(host,[4 5 6 7 0 1 2 3]));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "When should I call `ifftshift`?",
            answer: "Call `ifftshift` right before `ifft` (or `ifftn`) after you have applied `fftshift`-based processing in the frequency domain. It restores the DC component to index 1.",
        },
        BuiltinDocumentationFaq {
            question: "Is `ifftshift` always the inverse of `fftshift`?",
            answer: "Yes. For any supported input, `ifftshift(fftshift(X))` returns `X`, including odd-length dimensions where the shift counts differ.",
        },
        BuiltinDocumentationFaq {
            question: "Does `ifftshift` modify data values?",
            answer: "No. It only reorders elements. Magnitudes, phases, and overall content stay the same.",
        },
        BuiltinDocumentationFaq {
            question: "Can I restrict `ifftshift` to specific axes?",
            answer: "Yes. Pass a dimension index, vector of indices, or logical mask exactly like in MATLAB.",
        },
        BuiltinDocumentationFaq {
            question: "Does `ifftshift` support gpuArray inputs?",
            answer: "Yes. RunMat keeps GPU data on-device whenever possible and falls back to a single gather/upload cycle otherwise.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "fftshift",
            target: BuiltinDocumentationLinkTarget::Builtin("fftshift"),
        },
        BuiltinDocumentationLink {
            label: "fft",
            target: BuiltinDocumentationLinkTarget::Builtin("fft"),
        },
        BuiltinDocumentationLink {
            label: "ifft",
            target: BuiltinDocumentationLinkTarget::Builtin("ifft"),
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
            label: "`crates/runmat-runtime/src/builtins/math/fft/ifftshift.rs`",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/fft/ifftshift.rs",
            ),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Runtime FFT implementation",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/fft/ifftshift.rs",
            ),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Runtime unit cases",
                location: "builtins::math::fft::ifftshift::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Runtime integration case",
                location: "builtins::math::fft::ifftshift::tests::ifftshift_gpu_roundtrip",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: None,
};
