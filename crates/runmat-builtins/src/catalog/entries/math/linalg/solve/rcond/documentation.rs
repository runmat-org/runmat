use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Inputs must be square matrices (scalars count as `1 × 1`). Non-square inputs trigger the error `\"rcond: input must be a square matrix.\"`",
            "For the empty matrix (`0 × 0`) the reciprocal condition number is `Inf`.",
            "`rcond(0)` returns `0`. Non-zero scalars return `1`.",
            "Public MATLAB matrix classes are single and double. RunMat mode separately admits logical matrices and all eight real integer classes through named extensions; integer values cross only after exact binary64 representability checks.",
            "Complex matrices use the same 1-norm estimator with adjoint solves; host Single input retains Single output class.",
            "Host execution estimates the reciprocal 1-norm condition from pivoted LU solves without forming an inverse matrix.",
            "Resident integer input bypasses floating provider condition-estimation kernels, gathers through its exact owner, and restores the double result to that owner when supported."
        ]
    },
    BuiltinDocumentationSection {
        heading: "Acceleration and residency",
        paragraphs: &[
            "When a provider exposes the `rcond` hook, the shipping backends gather to the host, use the shared 1-norm estimator, and restore the scalar to the source class when possible.",
            "When that hook is unavailable, the Runtime gathers through the exact owner and runs the same 1-norm estimator; a `linsolve` 2-norm condition hint is not substituted.",
            "Explicit device placement requires a class-preserving resident result or a typed capability error; no output is silently relabelled."
        ]
    },
    BuiltinDocumentationSection {
        heading: "Implementation and validation",
        paragraphs: &[
            "Original GPU declarations: precisions f32, f64; elementwise false; reduction false; broadcasting none.",
            "Original fusion declarations: elementwise false; reduction false; max inputs 1; constants inline."
        ]
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Reciprocal condition of the identity matrix",
        program: "A = eye(3);\nrc = rcond(A)",
        display_output: Some("rc = 1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(rc == 1);",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Detecting a nearly singular matrix",
        program: "A = diag([1, 1e-8]);\nrc = rcond(A)",
        display_output: Some("rc = 1e-8"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(rc-1e-8) < 1e-14);",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Reciprocal condition of a singular matrix",
        program: "A = [1 2; 2 4];\nrc = rcond(A)",
        display_output: Some("rc = 0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(rc == 0);",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Reciprocal condition for a complex matrix",
        program: "A = [1+2i 0; 3i 2-1i];\nrc = rcond(A)",
        display_output: Some("rc ≈ 0.1824"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(rc-0.1823725421878943) < 1e-6);",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Handling the empty matrix",
        program: "rc = rcond([])",
        display_output: Some("rc = Inf"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isinf(rc));",
        },
    },
    BuiltinExample {
        id: "example-6",
        title: "Estimating conditioning on GPU data",
        program: "G = gpuArray([2 0; 0 0.5]);\nrc_gpu = rcond(G);\nrc = gather(rc_gpu)",
        display_output: Some("rc = 0.25"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(rc-0.25) < 1e-10);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "What range of values should I expect from `rcond`?",
        answer: "`rcond` returns a value in `[0, 1]`. Values near `1` indicate a well-conditioned matrix. Values near `0` indicate singular or ill-conditioned matrices."
    },
    BuiltinDocumentationFaq {
        question: "Why does `rcond` return `Inf` for the empty matrix?",
        answer: "The empty matrix is perfectly conditioned by convention, so the reciprocal condition number is `Inf` (`1 / 0`)."
    },
    BuiltinDocumentationFaq {
        question: "Does `rcond` warn me about singular systems automatically?",
        answer: "`rcond` itself does not issue warnings. Use the value to decide whether the matrix is singular to working precision, similar to MATLAB workflows (`if rcond(A) < eps, ...`)."
    },
    BuiltinDocumentationFaq {
        question: "How is the reciprocal condition estimated?",
        answer: "RunMat uses a pivoted LU factorization and a bounded inverse 1-norm estimate, without forming the inverse matrix. This is separate from the SVD tolerance used by `pinv` and `rank`."
    },
    BuiltinDocumentationFaq {
        question: "Will calling `rcond` move my data off the GPU?",
        answer: "The shipping providers gather to the shared host estimator and restore a class-preserving scalar. A future provider may implement the same 1-norm contract directly."
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "inv",
        target: BuiltinDocumentationLinkTarget::Builtin("inv"),
    },
    BuiltinDocumentationLink {
        label: "pinv",
        target: BuiltinDocumentationLinkTarget::Builtin("pinv"),
    },
    BuiltinDocumentationLink {
        label: "rank",
        target: BuiltinDocumentationLinkTarget::Builtin("rank"),
    },
    BuiltinDocumentationLink {
        label: "det",
        target: BuiltinDocumentationLinkTarget::Builtin("det"),
    },
    BuiltinDocumentationLink {
        label: "linsolve",
        target: BuiltinDocumentationLinkTarget::Builtin("linsolve"),
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
        label: "cond",
        target: BuiltinDocumentationLinkTarget::Builtin("cond"),
    },
    BuiltinDocumentationLink {
        label: "norm",
        target: BuiltinDocumentationLinkTarget::Builtin("norm"),
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("rcond"),
    slug: Some("rcond"),
    summary: "Estimate reciprocal condition numbers in MATLAB and RunMat.",
    description: "`rc = rcond(A)` estimates the reciprocal 1-norm condition number of square matrix `A`, preserving the Single class for Single input.",
    keywords: &[
        "rcond",
        "reciprocal condition",
        "condition number",
        "linear algebra",
        "gpu"
    ],
    related: &[
        "inv",
        "pinv",
        "rank",
        "det",
        "linsolve",
        "gpuarray",
        "gather",
        "cond",
        "norm"
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
                label: "crates/runmat-runtime/src/builtins/math/linalg/solve/rcond.rs",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/solve/rcond.rs")
            }
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::math::linalg::solve::rcond::tests"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::ProviderTest,
                label: "gpu",
                location: "builtins::math::linalg::solve::rcond::tests::rcond_gpu_round_trip_matches_cpu"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::WgpuTest,
                label: "wgpu",
                location: "builtins::math::linalg::solve::rcond::tests::rcond_wgpu_matches_cpu"
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
