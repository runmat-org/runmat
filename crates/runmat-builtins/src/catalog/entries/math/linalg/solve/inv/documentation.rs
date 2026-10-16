use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Inputs must be 2-D matrices (trailing singleton dimensions are accepted). Non-square matrices raise the MATLAB error `\"inv: input must be a square matrix.\"`",
            "Singular or rank-deficient matrices raise `\"inv: matrix is singular to working precision.\"`",
            "Logical and integer inputs are promoted to double precision before inversion.",
            "Complex inputs are handled in full complex arithmetic.",
            "Empty matrices return an empty matrix with the same dimensions (e.g., `inv([])` yields `[]`)."
        ]
    },
    BuiltinDocumentationSection {
        heading: "Acceleration and residency",
        paragraphs: &[
            "Uses the acceleration provider's `inv` hook when available; the default WGPU backend gathers to the host, computes the inverse, and re-uploads the result to preserve residency.",
            "When a GPU acceleration provider is active, RunMat forwards the operation to its `inv` hook. If the provider does not implement a native kernel, RunMat gathers the data to the host, uses the shared CPU implementation, and attempts to re-upload the result so downstream GPU work keeps its residency. The shipping WGPU backend currently follows this gather/compute/upload pattern.",
            "You typically do not need to move data manually. If `A` already resides on the GPU and the provider implements `inv`, the computation stays on the device. Providers without a native kernel (including the current WGPU backend) download `A`, compute the inverse on the host, and re-upload the result, so subsequent GPU code continues to operate on device-resident data. `gpuArray` remains available for compatibility and for explicitly seeding GPU residency."
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
        title: "Inverting a 2x2 matrix for solving linear systems",
        program: "A = [4 -2; 1 3];\nX = inv(A)",
        display_output: Some("X = [3/14, 1/7; -1/14, 2/7]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(norm(A*X-eye(2)) < 1e-10);"
        }
    },
    BuiltinExample {
        id: "example-2",
        title: "Checking that `inv(A)` produces the identity matrix",
        program: "A = [2 1 0; 0 1 -1; 0 0 3];\nX = inv(A);\nproduct = A * X",
        display_output: Some("product =\n    1.0000         0         0\n         0    1.0000         0\n         0         0    1.0000"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(norm(product-eye(3)) < 1e-10);"
        }
    },
    BuiltinExample {
        id: "example-3",
        title: "Inverting a diagonal matrix with symbolic structure",
        program: "D = diag([2, 5, 10]);\nX = inv(D)",
        display_output: Some("X =\n    0.5000         0         0\n         0    0.2000         0\n         0         0    0.1000"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(norm(D*X-eye(3)) < 1e-10);"
        }
    },
    BuiltinExample {
        id: "example-4",
        title: "Computing the inverse of a complex matrix",
        program: "A = [1+2i  0; 3i  4-1i];\nX = inv(A)",
        display_output: Some("X ≈ [0.2000-0.4000i, 0; -0.2471-0.2118i, 0.2353+0.0588i]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(norm(A*X-eye(2)) < 1e-10);"
        }
    },
    BuiltinExample {
        id: "example-5",
        title: "Using `inv` on a GPU-resident matrix",
        program: "G = gpuArray([3 1; 0 2]);\ninvG = inv(G);       % stays on the GPU when the provider implements inv\nresult = gather(invG)",
        display_output: Some("result =\n    0.3333   -0.1667\n         0    0.5000"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(norm(result*[3 1; 0 2]-eye(2)) < 1e-10);"
        }
    },
    BuiltinExample {
        id: "example-6",
        title: "Handling singular matrices gracefully",
        program: "A = [1 2; 2 4];\nX = inv(A)",
        display_output: Some("Error using inv\ninv: matrix is singular to working precision."),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::ExpectedError {
            identifier: "MATLAB:inv:InvalidInput"
        }
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Do I need to use `inv` to solve linear systems?",
        answer: "Prefer `mldivide` (`A \\\\ b`) or `linsolve` for numerical stability and performance. Use `inv` only when you explicitly need the inverse matrix."
    },
    BuiltinDocumentationFaq {
        question: "What error do I get for singular matrices?",
        answer: "RunMat mirrors MATLAB and raises `\"inv: matrix is singular to working precision.\"` when LU factorisation detects a zero pivot."
    },
    BuiltinDocumentationFaq {
        question: "Can I invert non-square matrices?",
        answer: "No. `inv` requires square matrices. Use `pinv` for pseudoinverses of rectangular matrices."
    },
    BuiltinDocumentationFaq {
        question: "Does `inv` support complex numbers?",
        answer: "Yes. Complex matrices are inverted using full complex arithmetic."
    },
    BuiltinDocumentationFaq {
        question: "What happens with empty matrices?",
        answer: "`inv([])` returns `[]` (an empty matrix) without error."
    },
    BuiltinDocumentationFaq {
        question: "Does `inv` preserve GPU residency?",
        answer: "If the acceleration provider exposes an `inv` hook, the operation stays on the GPU. Otherwise, RunMat gathers, computes on the host, and re-uploads so the caller still receives a GPU tensor."
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "pinv",
        target: BuiltinDocumentationLinkTarget::Builtin("pinv"),
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
        label: "det",
        target: BuiltinDocumentationLinkTarget::Builtin("det"),
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
    title: Some("inv"),
    slug: Some("inv"),
    summary: "Compute inverses of square matrices in MATLAB and RunMat.",
    description: "`X = inv(A)` returns the matrix inverse of square, full-rank matrix `A`. Scalar inputs behave like `1 ./ A`, and decomposition/pivoting behavior follows MATLAB semantics.",
    keywords: &[
        "inv",
        "matrix inverse",
        "linear algebra",
        "solve",
        "gpu"
    ],
    related: &[
        "pinv",
        "linsolve",
        "mldivide",
        "det",
        "gpuarray",
        "gather",
        "cond",
        "norm",
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
                label: "`crates/runmat-runtime/src/builtins/math/linalg/solve/inv.rs`",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/solve/inv.rs")
            }
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::math::linalg::solve::inv::tests"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::ProviderTest,
                label: "gpu",
                location: "builtins::math::linalg::solve::inv::tests::inv_gpu_round_trip_matches_cpu"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::WgpuTest,
                label: "wgpu",
                location: "builtins::math::linalg::solve::inv::tests::inv_wgpu_matches_cpu"
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
