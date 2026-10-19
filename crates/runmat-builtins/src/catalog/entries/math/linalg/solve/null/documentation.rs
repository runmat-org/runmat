use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Inputs must behave like 2-D matrices. Trailing singleton dimensions are accepted; other higher-rank inputs raise `RunMat:null:InvalidInput`.",
            "`null(A)` computes the default rank tolerance from the singular values as `max(size(A)) * eps(max(svd(A)))`, then returns an orthonormal basis for singular directions at or below that cutoff.",
            "`null(A, tol)` uses `tol` as the rank threshold and returns the corresponding orthonormal null-space basis.",
            "`null(A, \"rational\")` accepts either char or string scalar options and returns the free-variable basis derived from reduced row echelon form; `\"r\"` is retained as a legacy alias.",
            "The documented matrix domain is single or double. Native integer and logical matrices are separately gated RunMat extensions; integer values must be exactly representable before the double SVD or row-reduction boundary.",
            "A native integer tolerance is a documented real numeric scalar control and must be exactly representable before the floating rank comparison.",
            "Real and complex single inputs return single bases. Double and accepted integer/logical inputs return double bases.",
            "Real and complex orthonormal bases use SVD rank decisions and right singular vectors where available; wide matrices complete the orthogonal complement so the full nullity is returned even when the underlying SVD implementation is thin.",
            "Complex inputs return complex null-space bases and use complex row reduction for the `'r'` option.",
            "Full-rank matrices return an empty `n x 0` basis where `n` is the input column count. A `0 x n` matrix returns an `n x n` identity basis."
        ]
    },
    BuiltinDocumentationSection {
        heading: "Acceleration and residency",
        paragraphs: &[
            "Real floating GPU inputs are gathered through their exact owning provider and the basis is restored to that same owner with its computed floating class and source provenance. The rational basis form rejects GPU input before fallback.",
            "`null` is an eager solve-style operation. It terminates fusion plans and gathers supported GPU-resident inputs for null-space evaluation without consulting an unrelated ambient provider.",
            "The orthonormal form gathers a real gpuArray non-destructively through its exact owner and restores a class-correct basis to that owner. Automatic residency may fall back to a host value when the computed floating class cannot be represented; explicit gpuArray intent is terminal instead of being silently lost."
        ]
    },
    BuiltinDocumentationSection {
        heading: "Implementation and validation",
        paragraphs: &[
            "Original GPU declarations: precisions f32, f64; elementwise false; reduction false; broadcasting none.",
            "Original fusion declarations: elementwise false; reduction false; max inputs 2; constants inline."
        ]
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Orthonormal null-space basis",
        program: "A = [1 2; 2 4];\nZ = null(A)",
        display_output: Some("Z is a 2-by-1 orthonormal vector and A*Z is approximately zero"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(Z),[2 1])); assert(norm(A*Z) < 1e-10);",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Row-reduction basis",
        program: "A = [1 2 3; 2 4 6];\nZ = null(A, \"rational\")",
        display_output: Some("Z = [-2 -3; 1 0; 0 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(Z),[3 2])); assert(norm(A*Z) < 1e-10);",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Using a custom tolerance",
        program: "A = diag([1, 1e-8]);\nZ = null(A, 1e-6)",
        display_output: Some("Z = [0; 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(Z),[2 1])); assert(norm(A*Z) < 1e-6);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "What is the difference between `null(A)` and `null(A, \"rational\")`?",
        answer: "`null(A)` returns an orthonormal basis. `null(A, \"rational\")` returns a row-reduction basis where free variables are set to identity columns, so the result is often easier to read for simple dependent systems."
    },
    BuiltinDocumentationFaq {
        question: "Does `null` preserve GPU residency?",
        answer: "For the orthonormal form, RunMat gathers a real gpuArray through its exact owner and restores the result to that owner with matching explicit or automatic provenance. The rational form does not support gpuArray input."
    },
    BuiltinDocumentationFaq {
        question: "How is the default tolerance chosen?",
        answer: "RunMat follows the standard singular-value tolerance shape used by MATLAB-style rank decisions: `max(size(A)) * eps(max(svd(A)))`."
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "rref",
        target: BuiltinDocumentationLinkTarget::Builtin("rref"),
    },
    BuiltinDocumentationLink {
        label: "rank",
        target: BuiltinDocumentationLinkTarget::Builtin("rank"),
    },
    BuiltinDocumentationLink {
        label: "pinv",
        target: BuiltinDocumentationLinkTarget::Builtin("pinv"),
    },
    BuiltinDocumentationLink {
        label: "svd",
        target: BuiltinDocumentationLinkTarget::Builtin("svd"),
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("null"),
    slug: Some("null"),
    summary: "Compute a basis for a matrix null space.",
    description: "`null(A)` returns an orthonormal basis for the null space of a real or complex numeric matrix. `null(A, tol)` uses a caller-provided finite, non-negative rank tolerance. `null(A, \"rational\")` returns a row-reduction basis useful for simple exact-looking dependencies; RunMat also accepts `\"r\"` as a legacy alias.",
    keywords: &[
        "null",
        "null space",
        "kernel",
        "rank",
        "svd",
        "rref",
        "orthonormal"
    ],
    related: &[
        "rref",
        "rank",
        "pinv",
        "svd"
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
                label: "`crates/runmat-runtime/src/builtins/math/linalg/solve/null.rs`",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/solve/null.rs")
            }
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::math::linalg::solve::null::tests"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "integration",
                location: "core::tests::null_row_reduction_builtin_uses_semantic_vm"
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
