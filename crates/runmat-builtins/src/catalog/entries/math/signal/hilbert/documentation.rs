use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("hilbert"),
    slug: Some("hilbert"),
    summary: "Construct the complex analytic signal of a real signal.",
    description: "`z = hilbert(x)` transforms the real part of `x` along its first nonsingleton dimension. `z = hilbert(x,N)` first pads or truncates that dimension to length `N`. The result is complex, with its real part equal to the transformed real input.",
    keywords: &[
        "hilbert",
        "analytic signal",
        "quadrature",
        "FFT",
        "complex",
        "GPU",
    ],
    related: &["fft", "ifft", "angle", "unwrap", "filter"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Transform and shape",
            paragraphs: &[
                "The first nonsingleton dimension is transformed: each column of a matrix independently, or along a row vector. `N` changes only that dimension. `[]` leaves the original length.",
                "The FFT-domain multiplier retains DC and the even-length Nyquist bin, doubles strictly positive frequencies, and removes negative frequencies before inverse FFT.",
                "For complex single or double input, the imaginary component is ignored. The output remains single-complex for single input and double-complex for double input.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Compatibility and placement",
            paragraphs: &[
                "Integer and logical signal data, typed or logical lengths, zero or empty lengths, and N-D input beyond matrices require RunMat extension mode. Integer samples must convert exactly to binary64; typed complex-integer input is not supported.",
                "A real resident floating signal can use its owning provider's `signal_hilbert` operation and return a distinct complex-interleaved resident output. Unsupported provider forms download non-destructively through the owner, compute on the host, and restore to that owner when its precision is supported. Provider execution or malformed-output errors are terminal.",
                "Hilbert is an FFT-domain fusion boundary. Runtime placement and provider capabilities decide where execution occurs; no GPU route is guaranteed for every form.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "quadrature-cosine",
            title: "Build a quadrature cosine",
            program: "x = [1 0 -1 0]; z = hilbert(x)",
            display_output: Some(
                "z = [1.0000 + 0.0000i, 0.0000 + 1.0000i, -1.0000 + 0.0000i, 0.0000 - 1.0000i]",
            ),
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(max(abs(z-[1, 1i, -1, -1i])) < 1e-10);",
            },
        },
        BuiltinExample {
            id: "length",
            title: "Select an FFT length",
            program: "x = [1 0 -1 0]; z = hilbert(x,8); size(z)",
            display_output: Some("ans = [1 8]"),
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(z),[1 8]));",
            },
        },
        BuiltinExample {
            id: "matrix-columns",
            title: "Transform matrix columns",
            program: "X = [1 0;0 1;-1 0;0 -1]; Z = hilbert(X)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(Z),[4 2])); assert(max(abs(Z(:,1)-[1;1i;-1;-1i])) < 1e-10);",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "What does `hilbert` return?",
            answer: "A complex analytic signal whose real part follows the real input and whose imaginary part is its Hilbert transform.",
        },
        BuiltinDocumentationFaq {
            question: "Which dimension is transformed?",
            answer: "The first nonsingleton dimension: columns for matrices and the second dimension for row vectors.",
        },
        BuiltinDocumentationFaq {
            question: "What happens to complex input?",
            answer: "The imaginary input component is ignored; the real part is transformed.",
        },
        BuiltinDocumentationFaq {
            question: "Does the result remain on the GPU?",
            answer: "When the input owner supports `signal_hilbert`, yes. Unsupported forms can use an owner-preserving host fallback and restore; operational failures are not silently retried.",
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
            label: "filter",
            target: BuiltinDocumentationLinkTarget::Builtin("filter"),
        },
        BuiltinDocumentationLink {
            label: "Source",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/hilbert.rs",
            ),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Runtime",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/hilbert.rs",
            ),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Analytic signal, compatibility, and provider ownership",
            location: "builtins::math::signal::hilbert::tests",
        }],
        notes: &[
            "Resident tests need an available provider. The portable examples exercise the host reference path.",
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
