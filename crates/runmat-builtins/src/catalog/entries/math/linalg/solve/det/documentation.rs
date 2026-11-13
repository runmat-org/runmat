use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Inputs must be 2-D square matrices. RunMat rejects non-square or higher-rank inputs with the MATLAB error `\"det: input must be a square matrix.\"`",
            "Determinants are computed from an LU factorization with partial pivoting, ensuring numerical behavior that mirrors MATLAB. Singular matrices return `0` (within floating-point round-off).",
            "MATLAB documents single and double matrices. Integer and logical matrices are separately gated RunMat-only extensions; integer values must be exactly representable as double rather than silently rounded.",
            "Single input produces a single result. Complex input retains complex result storage and its single/double class.",
            "The determinant of the empty matrix (`[]`) is `1`."
        ]
    },
    BuiltinDocumentationSection {
        heading: "Acceleration and residency",
        paragraphs: &[
            "Uses owner-resolved provider LU for floating storage. Typed integer, logical, and complex storage use an explicit gathered fallback; restoration requires the owner to physically support the computed precision, and results are validated against input owner/device and computed shape/storage/precision.",
            "RunMat resolves the provider from the input handle rather than a global provider. Floating matrices use its `lu` hook when possible; typed integer, logical, complex, and unsupported-provider cases gather explicitly. Re-uploaded results must match the input owner/device and the computed shape, storage, and precision or they are freed and rejected.",
            "You usually do NOT need to call `gpuArray` yourself in RunMat. Floating storage uses the owning provider's `lu` hook when available. Explicit fallback paths gather through that owner and restore real or complex scalar results to the same device while preserving single/double precision."
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
        title: "Determinant Of A 2x2 Matrix",
        program: "A = [4 -2; 1 3];\nd = det(A)",
        display_output: Some("d = 14"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(d == 14);"
        }
    },
    BuiltinExample {
        id: "example-2",
        title: "Checking Zero Determinant For Singular Matrix",
        program: "B = [1 2; 2 4];\nd = det(B)",
        display_output: Some("d = 0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(d == 0);"
        }
    },
    BuiltinExample {
        id: "example-3",
        title: "Determinant Of A Complex Matrix",
        program: "C = [1+2i 0; 3i 4-1i];\nd = det(C)",
        display_output: Some("d = 6 + 7i"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(d - (6+7i)) < 1e-12);"
        }
    },
    BuiltinExample {
        id: "example-4",
        title: "Determinant Equals Product Of Diagonal For Triangular Matrix",
        program: "U = [3 2 1; 0 5 -1; 0 0 2];\nd = det(U)",
        display_output: Some("d = 30"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(d == 30);"
        }
    },
    BuiltinExample {
        id: "example-5",
        title: "Determinant Of An Empty Matrix",
        program: "d = det([])",
        display_output: Some("d = 1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(d == 1);"
        }
    },
    BuiltinExample {
        id: "example-6",
        title: "Using `det` With `gpuArray` Data",
        program: "G = gpuArray([2 0 0; 0 3 0; 1 0 4]);\nd_gpu = det(G);     % stays on the GPU when a provider is active\nd = gather(d_gpu)",
        display_output: Some("d = 24"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(d == 24);"
        }
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Why does `det` require square matrices?",
        answer: "The determinant is only defined for square matrices. RunMat mirrors MATLAB by rejecting non-square inputs with `\"det: input must be a square matrix.\"`"
    },
    BuiltinDocumentationFaq {
        question: "What is `det([])`?",
        answer: "The determinant of the empty matrix (`[]`) is `1`, matching MATLAB's convention for the product of an empty diagonal."
    },
    BuiltinDocumentationFaq {
        question: "Does `det` warn for nearly singular matrices?",
        answer: "No. RunMat returns the floating-point result of the LU-based determinant. Very small magnitudes indicate near singularity, just as in MATLAB."
    },
    BuiltinDocumentationFaq {
        question: "How does `det` handle complex matrices?",
        answer: "Complex matrices are factorized in complex arithmetic. The result is complex; if the imaginary part happens to be zero, it is still reported through the complex number interface."
    },
    BuiltinDocumentationFaq {
        question: "Will the result stay on the GPU?",
        answer: "Yes, when a GPU provider is active RunMat re-uploads the scalar determinant so that subsequent GPU code continues to operate on device-resident data. Providers without LU support fall back to the CPU path and return a host scalar."
    },
    BuiltinDocumentationFaq {
        question: "Can `det` overflow or underflow?",
        answer: "Large determinants can overflow or underflow double precision just as in MATLAB. RunMat does not rescale the matrix; if you require scaling, consider `log(det(A))` via `lu` or `chol`."
    },
    BuiltinDocumentationFaq {
        question: "Do logical inputs work?",
        answer: "Logical matrices are a named RunMat-only extension. Strict MATLAB compatibility mode rejects them before data access."
    },
    BuiltinDocumentationFaq {
        question: "Is the determinant computed exactly?",
        answer: "No. The LU factorization works in floating-point, so the result is subject to round-off. The behavior matches MATLAB's `det`."
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
        label: "lu",
        target: BuiltinDocumentationLinkTarget::Builtin("lu"),
    },
    BuiltinDocumentationLink {
        label: "trace",
        target: BuiltinDocumentationLinkTarget::Builtin("trace"),
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
        label: "linsolve",
        target: BuiltinDocumentationLinkTarget::Builtin("linsolve"),
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
    title: Some("det"),
    slug: Some("det"),
    summary: "Compute determinants of square matrices in MATLAB and RunMat.",
    description: "`det(A)` returns the determinant of real or complex square matrix `A`. Scalar and empty-matrix cases (`det([]) = 1`) follow MATLAB and RunMat conventions.",
    keywords: &[
        "det",
        "determinant",
        "linear algebra",
        "matrix",
        "gpu"
    ],
    related: &[
        "inv",
        "pinv",
        "lu",
        "trace",
        "gpuarray",
        "gather",
        "cond",
        "linsolve",
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
                label: "`crates/runmat-runtime/src/builtins/math/linalg/solve/det.rs`",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/solve/det.rs")
            }
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::math::linalg::solve::det::tests"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::ProviderTest,
                label: "gpu",
                location: "builtins::math::linalg::solve::det::tests::det_rejects_aliased_or_mistyped_provider_lu_results_and_frees_safely"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::WgpuTest,
                label: "wgpu",
                location: "builtins::math::linalg::solve::det::tests::det_wgpu_matches_cpu"
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
