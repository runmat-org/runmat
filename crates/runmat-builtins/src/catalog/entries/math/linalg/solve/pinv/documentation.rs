use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Supports scalars, vectors, and higher-dimensional inputs that behave like matrices (trailing singleton dimensions are allowed; other higher ranks must be reshaped first).",
            "Public MATLAB matrix classes are single and double. RunMat mode separately admits logical matrices and all eight real integer classes through named extensions; integer data must be exactly representable as binary64, the result is double, and wider values reject instead of rounding.",
            "Optional second argument `pinv(A, tol)` treats values in `tol` as the user-specified cutoff for singular values. Entries ≤ `tol` contribute zeros in the diagonal of `Σ⁺`.",
            "Complex matrices are handled in full complex arithmetic via `A = U * Σ * Vᴴ`.",
            "Empty matrices return the appropriately sized zero matrix (`size(pinv(A)) == fliplr(size(A))`).",
            "The result always has size `n × m` if the input is `m × n`.",
            "RunMat preserves single or double floating class in the result, including complex inputs; its integer and logical input extensions produce double. Nonfinite inputs produce NaN-valued results instead of entering the SVD solver.",
            "A typed-integer tolerance is a separate RunMat extension because the public scalar tolerance does not enumerate integer classes. It must be exactly representable as binary64 before threshold computation; logical tolerance is independently gated.",
            "Resident integer input bypasses floating provider pseudoinverse kernels, gathers through its exact owner, and restores the double result to that owner when supported."
        ]
    },
    BuiltinDocumentationSection {
        heading: "Acceleration and residency",
        paragraphs: &[
            "Invokes the acceleration provider's pinv hook when available; the current WGPU backend gathers to the host, runs the shared SVD implementation, and re-uploads the result to keep downstream residency intact.",
            "When a GPU acceleration provider is active, RunMat offers the operation through its `pinv` hook, passing along any explicit tolerance. Providers may implement a native GPU kernel; otherwise, they can gather to the host, invoke the shared SVD routine, and re-upload the result. The shipping WGPU backend follows this gather/compute/upload pattern today, so downstream GPU work retains residency without MATLAB-level changes.",
            "Explicit residency management is rarely required. When inputs already live on the GPU and the provider implements `pinv`, the builtin executes entirely on the device. Providers without a native kernel (including the current WGPU backend) transparently download the matrix, run the shared CPU path, and re-upload the result so the caller continues working with a GPU tensor. `gpuArray` remains available for MATLAB compatibility or to seed GPU residency explicitly."
        ]
    },
    BuiltinDocumentationSection {
        heading: "Implementation and validation",
        paragraphs: &[
            "Original GPU declarations: precisions f32, f64; elementwise false; reduction false; broadcasting none.",
            "Original fusion declarations: elementwise false; reduction false; max inputs 1; constants uniform."
        ]
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Finding the pseudoinverse of a tall matrix",
        program: "A = [1 0; 0 0; 0 1];\nX = pinv(A)",
        display_output: Some("X =\n     1     0     0\n     0     0     1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(norm(A*X*A-A) < 1e-10);",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Solving an overdetermined least-squares problem",
        program: "A = [1 1; 1 2; 1 3];\nb = [1; 0; 0];\nx = pinv(A) * b",
        display_output: Some("x = [1.3333; -0.5000]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(norm(A*x-b) < 1); assert(norm(x-[4/3;-1/2]) < 1e-4);",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Suppressing small singular values with a custom tolerance",
        program: "A = diag([1, 1e-10]);\nX = pinv(A, 1e-6)",
        display_output: Some("X =\n     1     0\n     0     0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(norm(X-diag([1 0])) < 1e-10);",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Pseudoinverse of a rank-deficient square matrix",
        program: "A = [1 2; 2 4];\nX = pinv(A)",
        display_output: Some("X =\n    0.0400    0.0800\n    0.0800    0.1600"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(norm(A*X*A-A) < 1e-10);",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Pseudoinverse of a complex diagonal matrix",
        program: "A = diag([2+1i, 3-2i]);\nX = pinv(A)",
        display_output: Some("X =\n   0.4000 - 0.2000i         0\n         0   0.2308 + 0.1538i"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(norm(A*X*A-A) < 1e-10);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "How is `pinv` different from `inv`?",
        answer: "`inv(A)` requires `A` to be square and full rank. `pinv(A)` works for any matrix shape and produces the minimum-norm solution even when `A` is singular or rectangular."
    },
    BuiltinDocumentationFaq {
        question: "What tolerance does `pinv` use by default?",
        answer: "RunMat matches MATLAB: `max(size(A)) * eps(max(s))`, where `s` are the singular values. Values below this threshold are treated as zero when forming `Σ⁺`."
    },
    BuiltinDocumentationFaq {
        question: "Can I recover the rank from the pseudoinverse?",
        answer: "Yes. Count the singular values greater than the effective tolerance (`rank` returns this directly). The same tolerance drives both `pinv` and `rank`."
    },
    BuiltinDocumentationFaq {
        question: "Does `pinv` support complex inputs?",
        answer: "Absolutely. Complex matrices use a full complex SVD with conjugate transposes, matching MATLAB's definition."
    },
    BuiltinDocumentationFaq {
        question: "Will calling `pinv` move my data off the GPU?",
        answer: "Only if the active provider lacks a native implementation. In that case RunMat gathers, computes, and re-uploads for you. Providers may expose native kernels to keep the entire computation on the device."
    },
    BuiltinDocumentationFaq {
        question: "Is `pinv(A) * b` equivalent to `A \\\\ b`?",
        answer: "For full-rank systems, yes. `A \\\\ b` is typically faster and more numerically stable, but `pinv` remains useful for ill-conditioned or rank-deficient problems where the pseudoinverse is desired."
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "inv",
        target: BuiltinDocumentationLinkTarget::Builtin("inv"),
    },
    BuiltinDocumentationLink {
        label: "linsolve",
        target: BuiltinDocumentationLinkTarget::Builtin("linsolve"),
    },
    BuiltinDocumentationLink {
        label: "mldivide",
        target: BuiltinDocumentationLinkTarget::Builtin("mldivide"),
    },
    BuiltinDocumentationLink {
        label: "mrdivide",
        target: BuiltinDocumentationLinkTarget::Builtin("mrdivide"),
    },
    BuiltinDocumentationLink {
        label: "svd",
        target: BuiltinDocumentationLinkTarget::Builtin("svd"),
    },
    BuiltinDocumentationLink {
        label: "rank",
        target: BuiltinDocumentationLinkTarget::Builtin("rank"),
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
        label: "det",
        target: BuiltinDocumentationLinkTarget::Builtin("det"),
    },
    BuiltinDocumentationLink {
        label: "norm",
        target: BuiltinDocumentationLinkTarget::Builtin("norm"),
    },
    BuiltinDocumentationLink {
        label: "rcond",
        target: BuiltinDocumentationLinkTarget::Builtin("rcond"),
    },
    BuiltinDocumentationLink {
        label: "Open an issue",
        target: BuiltinDocumentationLinkTarget::External(
            "https://github.com/runmat-org/runmat/issues/new/choose",
        ),
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("pinv"),
    slug: Some("pinv"),
    summary: "Compute Moore-Penrose pseudoinverses using SVD-based tolerance rules with MATLAB-compatible defaults.",
    description: "`pinv(A)` computes pseudoinverses from singular-value decompositions using MATLAB-compatible default tolerance rules, with optional explicit tolerance overrides.",
    keywords: &[
        "pinv",
        "pseudoinverse",
        "least squares",
        "svd",
        "gpu"
    ],
    related: &[
        "inv",
        "linsolve",
        "mldivide",
        "mrdivide",
        "svd",
        "rank",
        "gpuarray",
        "gather",
        "cond",
        "det",
        "norm",
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
                label: "`crates/runmat-runtime/src/builtins/math/linalg/solve/pinv.rs`",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/solve/pinv.rs")
            }
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::math::linalg::solve::pinv::tests"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::ProviderTest,
                label: "gpu",
                location: "builtins::math::linalg::solve::pinv::tests::pinv_gpu_round_trip_matches_cpu"
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
