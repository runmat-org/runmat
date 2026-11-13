use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Inputs must behave like 2-D matrices. Trailing singleton dimensions are accepted; other higher-rank inputs raise `RunMat:rref:InvalidInput`.",
            "The default tolerance is `max(size(A)) * eps(norm(A, inf))`, matching MATLAB row-reduction conventions.",
            "`rref(A, tol)` uses a caller-provided finite, non-negative scalar pivot tolerance.",
            "The MATLAB-compatible matrix and tolerance classes are single and double. RunMat mode additionally accepts integer matrices and integer tolerances only through explicit exact-binary64 checks; strict MATLAB mode rejects those extensions.",
            "Complex inputs use complex Gauss-Jordan elimination with pivot selection by complex magnitude.",
            "The second output is a row vector of one-based pivot column indices.",
            "Non-finite values are accepted. The default tolerance still follows `max(size(A)) * eps(norm(A, inf))`; `Inf` and `NaN` therefore follow the runtime's IEEE floating-point tolerance and pivot arithmetic."
        ]
    },
    BuiltinDocumentationSection {
        heading: "Acceleration and residency",
        paragraphs: &[
            "GPU inputs are gathered to the shared host row-reduction implementation. Real reduced matrices are re-uploaded when a provider is available; upload failures surface as `RunMat:rref:Internal`.",
            "`rref` is an eager solve-style operation. It terminates fusion plans, gathers GPU-resident inputs for numerical row reduction, and returns ordinary MATLAB-compatible values.",
            "When the input is a real gpuArray, RunMat gathers it, computes the row-reduced matrix on the host, and re-uploads the reduced matrix through that handle's owning provider. If owner-preserving re-upload fails, the builtin raises `RunMat:rref:Internal` rather than silently changing residency. Pivot-column metadata is returned as a host vector."
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
        title: "Finding dependent columns",
        program: "A = [1 2 3; 2 4 6; 1 1 1];\n[R, pivots] = rref(A)",
        display_output: Some("R = [1 0 -1; 0 1 2; 0 0 0]\npivots = [1 2]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(norm(R-[1 0 -1;0 1 2;0 0 0]) < 1e-10); assert(isequal(pivots,[1 2]));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Using a custom tolerance",
        program: "A = diag([1, 1e-12]);\nR = rref(A, 1e-6)",
        display_output: Some("R = [1 0; 0 0]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(norm(R-[1 0;0 0]) < 1e-10);",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Row-reducing an augmented system",
        program: "A = [2 -1 3; 4 -2 6; 1 0 1];\nb = [5; 10; 2];\nRAug = rref([A b])",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(RAug),[3 4])); assert(rank(RAug) == 2);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "How does `rref` choose pivots?",
        answer: "RunMat scans columns left to right, picks the largest available pivot by magnitude below the current pivot row, normalizes the pivot row, and eliminates the pivot column above and below the pivot."
    },
    BuiltinDocumentationFaq {
        question: "Is the pivot vector one-based?",
        answer: "Yes. The second output uses MATLAB's one-based column indices."
    },
    BuiltinDocumentationFaq {
        question: "Does `rref` use the same tolerance as `rank`?",
        answer: "No. `rank` uses an SVD tolerance. `rref` uses the MATLAB row-reduction tolerance based on the matrix infinity norm."
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "rank",
        target: BuiltinDocumentationLinkTarget::Builtin("rank"),
    },
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
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("rref"),
    slug: Some("rref"),
    summary: "Compute reduced row echelon form and pivot columns.",
    description: "`rref(A)` returns the reduced row echelon form of a real or complex numeric matrix. `[R, pivots] = rref(A)` also returns the one-based pivot column indices. An optional tolerance controls which entries are considered zero during pivot selection.",
    keywords: &[
        "rref",
        "reduced row echelon form",
        "row reduction",
        "pivot columns",
        "rank"
    ],
    related: &[
        "rank",
        "pinv",
        "linsolve",
        "mldivide"
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
                label: "`crates/runmat-runtime/src/builtins/math/linalg/solve/rref.rs`",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/solve/rref.rs")
            }
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::math::linalg::solve::rref::tests"
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "integration",
                location: "bytecode::compile::tests::compile_interprets_rref_rank_workflow"
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
