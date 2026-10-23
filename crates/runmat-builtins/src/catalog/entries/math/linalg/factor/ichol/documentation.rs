use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Syntax",
        paragraphs: &[
            "L = ichol(A)\nL = ichol(A, options)",
            "`A` must be a square sparse single- or double-precision real or complex matrix. The default computes a zero-fill incomplete lower Cholesky factor.",
            "`options` is a scalar struct. Supported fields are `type` (`'nofill'` or `'ict'`), `droptol` (nonnegative scalar), `michol` (`'off'` or `'on'`), `diagcomp` (nonnegative scalar), and `shape` (`'lower'` or `'upper'`).",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "With `type = 'nofill'`, the factor uses the selected triangular sparsity pattern of `A`. With `type = 'ict'`, fill entries in column `j` whose magnitude exceeds `droptol * norm(A(j:end,j),1)` over the selected triangle are retained.",
            "`michol = 'on'` compensates the diagonal for discarded contributions. `diagcomp` adds a nonnegative multiple of the source diagonal before factorization and can stabilize a matrix that otherwise encounters a nonpositive pivot.",
            "The default `shape = 'lower'` returns sparse `L`; `shape = 'upper'` returns its conjugate transpose.",
            "Input precision and real/complex class are preserved. A nonpositive pivot raises `RunMat:ichol:NotPositiveDefinite`.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution and ownership",
        paragraphs: &[
            "`ichol` borrows the input CSC arrays, admits all retained factor and workspace growth through the managed host allocation ledger, cooperatively checkpoints long work, and publishes a validated exact sparse output. It does not densify the matrix.",
            "Sparse factorization is a managed CPU operation. Resident or dense inputs are rejected rather than silently transferred or converted.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "nofill",
        title: "Zero-fill incomplete Cholesky factor",
        program: "A = sparse([4 -1 0; -1 4 -1; 0 -1 4]);\nL = ichol(A);\nassert(issparse(L));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(issparse(L)); assert(all(size(L) == [3 3]));",
        },
    },
    BuiltinExample {
        id: "threshold",
        title: "Threshold incomplete factorization",
        program: "A = sparse([6 -1 0; -1 6 -1; 0 -1 6]);\nopts = struct('type','ict','droptol',1e-3,'michol','on');\nL = ichol(A, opts);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(issparse(L)); assert(all(diag(L) > 0));",
        },
    },
    BuiltinExample {
        id: "upper",
        title: "Upper incomplete factor",
        program: "A = sparse([5 1; 1 3]);\nR = ichol(A, struct('shape','upper'));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(istriu(R)); assert(issparse(R));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "When should I use `nofill` or `ict`?",
        answer: "Use `nofill` for predictable storage bounded by the source pattern. Use `ict` when additional retained fill can improve the preconditioner; tune `droptol` to trade memory for quality.",
    },
    BuiltinDocumentationFaq {
        question: "What does `diagcomp` do?",
        answer: "It shifts each source diagonal before factorization. A small positive value can avoid breakdown for matrices whose zero-fill factor is not positive definite.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "chol",
        target: BuiltinDocumentationLinkTarget::Builtin("chol"),
    },
    BuiltinDocumentationLink {
        label: "pcg",
        target: BuiltinDocumentationLinkTarget::Builtin("pcg"),
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ichol"),
    slug: Some("ichol"),
    summary: "Compute sparse incomplete Cholesky factors without densifying.",
    description: "`ichol` constructs a managed sparse incomplete Cholesky factor for preconditioning and sparse iterative solves.",
    keywords: &["ichol", "incomplete Cholesky", "sparse", "preconditioner"],
    related: &["chol", "pcg"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "managed sparse ichol",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/factor/ichol.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "managed sparse ichol tests",
            location: "builtins::math::linalg::factor::ichol::tests",
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
