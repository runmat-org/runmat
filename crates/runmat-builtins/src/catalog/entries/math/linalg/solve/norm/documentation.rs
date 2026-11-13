use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Single input returns a single scalar tensor; double input returns double. RunMat integer and logical extensions return double.",
            "Typed-integer data remains authoritative until one checked binary64 norm boundary; values outside the exact binary64 integer range reject rather than round. If an owner cannot represent the required double output, automatic residency may finish on host while explicit gpuArray intent errors.",
            "Positive real scalar orders and the documented norm keywords are accepted subject to vector-versus-matrix restrictions. Logical order selectors, zero norm, and nuclear norm are separately gated RunMat extensions."
        ]
    },
    BuiltinDocumentationSection {
        heading: "Acceleration and residency",
        paragraphs: &[
            "Floating gpuArray input uses the exact owning provider's norm hook when available; validated host fallback restores a same-precision scalar to that owner. Integer and logical resident data are gated RunMat extensions and bypass floating provider hooks.",
            "Floating inputs use the exact owner's norm hook when it returns a fresh, correctly shaped and typed result. Unsupported hooks fall back to a non-consuming gather and class-preserving restoration. Integer and logical resident inputs are admitted only in RunMat mode and always take the checked host boundary instead of being mislabeled for a floating provider hook.",
            "```matlab:runnable\nG = gpuArray([3 4 12]);\nmag = norm(G);      % falls back to CPU today; future providers can stay on device\n```\nExpected output:\n```matlab\nmag = 13\n```"
        ]
    },
    BuiltinDocumentationSection {
        heading: "Implementation and validation",
        paragraphs: &[
            "Original GPU declarations: precisions f32, f64; elementwise false; reduction true; broadcasting none.",
            "Original fusion declarations: elementwise false; reduction false; max inputs 1; constants inline."
        ]
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Computing the Euclidean norm of a vector",
        program: "x = [3 4];\nmag = norm(x)",
        display_output: Some("mag = 5"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(mag == 5);"
        }
    },
    BuiltinExample {
        id: "example-2",
        title: "Calculating the Frobenius norm of a matrix",
        program: "A = [1 -2 3; -4 5 -6];\nf = norm(A, 'fro')",
        display_output: Some("f = 9.5394"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(f-sqrt(91)) < 1e-10);"
        }
    },
    BuiltinExample {
        id: "example-3",
        title: "Using the infinity norm for robust bounds",
        program: "x = [2 -7 4];\nbound = norm(x, Inf)",
        display_output: Some("bound = 7"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(bound == 7);"
        }
    },
    BuiltinExample {
        id: "example-4",
        title: "Summing singular values with the nuclear norm",
        program: "A = [2 0 0; 0 1 0];\ntau = norm(A, 'nuc')",
        display_output: Some("tau = 3"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tau == 3);"
        }
    },
    BuiltinExample {
        id: "example-5",
        title: "Computing the norm of complex-valued data",
        program: "z = [1+2i 3-4i];\nmag = norm(z)",
        display_output: Some("mag = 5.4772"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(mag-sqrt(30)) < 1e-10);"
        }
    },
    BuiltinExample {
        id: "example-6",
        title: "Preserving GPU residency transparently",
        program: "G = gpuArray([3 4 12]);\nmag = norm(G);      % falls back to CPU today; future providers can stay on device",
        display_output: Some("mag = 13"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(gather(mag)-13) < 1e-10);"
        }
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "What is the difference between `norm(x)` and `norm(x, 2)`?",
        answer: "No difference. Both compute the Euclidean norm for vectors (or the spectral norm for matrices)."
    },
    BuiltinDocumentationFaq {
        question: "Does `norm` support fractional powers?",
        answer: "Positive real scalar orders, including supported fractional vector orders, use the compatibility surface. Zero norm is available only when RunMat extensions are enabled; `Inf` and `-Inf` retain their documented meanings."
    },
    BuiltinDocumentationFaq {
        question: "Can I use `norm` on complex matrices?",
        answer: "Absolutely. RunMat mirrors MATLAB by computing singular values in complex arithmetic and always returning a non-negative real scalar."
    },
    BuiltinDocumentationFaq {
        question: "When should I use `'fro'` versus `'nuc'`?",
        answer: "`'fro'` is part of the compatibility surface and returns the square root of the sum of squares. `'nuc'` sums singular values but is a separately gated RunMat extension."
    },
    BuiltinDocumentationFaq {
        question: "Why does `norm` of an empty matrix return `0`?",
        answer: "That matches MATLAB’s convention: the sum over an empty set is zero, so every supported norm returns `0` for empty inputs."
    },
    BuiltinDocumentationFaq {
        question: "Can the norm overflow or underflow?",
        answer: "Yes. The result obeys IEEE-754 double precision rules, just like MATLAB. Extremely large or tiny values may overflow to `Inf` or underflow toward zero."
    },
    BuiltinDocumentationFaq {
        question: "Will the result stay on the GPU?",
        answer: "RunMat uses a validated provider norm hook for supported floating inputs. Otherwise it gathers without consuming the source handle and restores the same-precision scalar to the exact owner when that owner can represent it."
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "sum",
        target: BuiltinDocumentationLinkTarget::Builtin("sum"),
    },
    BuiltinDocumentationLink {
        label: "svd",
        target: BuiltinDocumentationLinkTarget::Builtin("svd"),
    },
    BuiltinDocumentationLink {
        label: "cond",
        target: BuiltinDocumentationLinkTarget::Builtin("cond"),
    },
    BuiltinDocumentationLink {
        label: "pinv",
        target: BuiltinDocumentationLinkTarget::Builtin("pinv"),
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
        label: "det",
        target: BuiltinDocumentationLinkTarget::Builtin("det"),
    },
    BuiltinDocumentationLink {
        label: "inv",
        target: BuiltinDocumentationLinkTarget::Builtin("inv"),
    },
    BuiltinDocumentationLink {
        label: "linsolve",
        target: BuiltinDocumentationLinkTarget::Builtin("linsolve"),
    },
    BuiltinDocumentationLink {
        label: "rank",
        target: BuiltinDocumentationLinkTarget::Builtin("rank"),
    },
    BuiltinDocumentationLink {
        label: "rcond",
        target: BuiltinDocumentationLinkTarget::Builtin("rcond"),
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("norm"),
    slug: Some("norm"),
    summary: "Compute vector and matrix norms in MATLAB and RunMat.",
    description: "`norm(X)` returns vector or matrix magnitudes. MATLAB-compatible data is single or double; all eight native integer classes and logical data are independently gated RunMat extensions.",
    keywords: &[
        "norm",
        "vector norm",
        "matrix norm",
        "frobenius",
        "nuclear",
        "gpu"
    ],
    related: &[
        "sum",
        "svd",
        "cond",
        "pinv",
        "gpuarray",
        "gather",
        "det",
        "inv",
        "linsolve",
        "rank",
        "rcond"
    ],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "`crates/runmat-runtime/src/builtins/math/linalg/solve/norm.rs`",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/solve/norm.rs")
            }
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::math::linalg::solve::norm::tests"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::ProviderTest,
                label: "gpu",
                location: "builtins::math::linalg::solve::norm::tests::norm_gpu_roundtrip_matches_cpu"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::WgpuTest,
                label: "wgpu",
                location: "builtins::math::linalg::solve::norm::tests::norm_wgpu_matches_cpu"
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
