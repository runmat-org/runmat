use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Syntax",
        paragraphs: &[
            "W = ilu(A)\n[L,U] = ilu(A)\n[L,U,P] = ilu(A)\n[___] = ilu(A, options)",
            "`A` is a square sparse single- or double-precision real or complex matrix. `W` equals `L + U - speye(size(A))`.",
            "Options fields are `type` (`'nofill'`, `'crout'`, or `'ilutp'`), nonnegative `droptol`, `milu` (`'off'`, `'row'`, or `'col'`), `udiag` (0 or 1), and `thresh` in [0,1].",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Modes and defaults",
        paragraphs: &[
            "`type = 'nofill'` is the default, retains only the input sparsity pattern, and ignores `droptol` and `thresh`. `type = 'crout'` admits threshold fill without pivoting. `type = 'ilutp'` applies threshold dropping and pivoting.",
            "`droptol` defaults to 0. For threshold modes, U uses `droptol * norm(A(:,j),2)` and L candidates are tested before pivot scaling. `thresh` is in `[0,1]`, defaults to 1, and is used by ILUTP.",
            "`milu` defaults to `'off'`; `'row'` or `'col'` compensates discarded contributions. `udiag` defaults to 0; in ILUTP, 1 replaces a zero pivot by its local drop tolerance.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Permutation, classes, and errors",
        paragraphs: &[
            "For `nofill` and `crout`, P is identity. With pivoted row-sum modification, `A*P` is approximated by `L*U`; with column-sum modification, `P*A` is approximated by `L*U`. When P is not requested its permutation is folded into U or L respectively.",
            "Input single/double precision and real/complex class are preserved. P is a sparse double permutation matrix. Dense, logical, integer, nonsquare, malformed, and nonfinite inputs are rejected.",
            "An unreplaced zero pivot raises `RunMat:ilu:NumericalFailure`; invalid options, invalid inputs, and managed runtime failures retain separate public identifiers.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution and ownership",
        paragraphs: &[
            "The implementation borrows validated CSC input, uses one claimed row workspace, claims every retained sparse growth before allocation, checkpoints cooperative work, and publishes validated exact sparse outputs. It never forms a dense matrix.",
            "Factor failures and cancellation are terminal after managed execution starts; no hidden dense or provider fallback is attempted.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "nofill",
        title: "Zero-fill sparse factors",
        program: "A = sparse([4 1 0; 2 5 1; 0 1 3]);\n[L,U] = ilu(A);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(issparse(L) && issparse(U)); assert(norm((A-L*U).*spones(A),'fro') < 1e-10);",
        },
    },
    BuiltinExample {
        id: "ilutp",
        title: "Threshold pivoted preconditioner",
        program: "A = sparse([1e-4 1; 1 2]);\no = struct('type','ilutp','droptol',1e-6,'thresh',1);\n[L,U,P] = ilu(A,o);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(issparse(P)); assert(norm(A*P-L*U,'fro') < 1e-3);",
        },
    },
    BuiltinExample {
        id: "combined",
        title: "Combined sparse preconditioner",
        program: "A = sparse([5 -1 0; -1 5 -1; 0 -1 5]);\nW = ilu(A);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(issparse(W)); assert(all(size(W) == size(A)));",
        },
    },
    BuiltinExample {
        id: "crout",
        title: "Crout threshold fill",
        program: "A = sparse([6 1 0; 1 7 1; 0 1 8]);\no = struct('type','crout','droptol',1e-3,'milu','row');\n[L,U] = ilu(A,o);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(issparse(L) && issparse(U)); assert(all(diag(L) == 1));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Which mode should I use?",
        answer: "Use nofill for source-pattern storage, crout for threshold fill without pivoting, and ilutp when pivoting is required.",
    },
    BuiltinDocumentationFaq {
        question: "What does droptol control?",
        answer: "Larger values discard more candidates and usually reduce storage at the cost of preconditioner quality. Nofill ignores droptol.",
    },
    BuiltinDocumentationFaq {
        question: "When should I request P?",
        answer: "Request P for the explicit ILUTP permutation identity. With two outputs it is folded into the appropriate factor.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "lu",
        target: BuiltinDocumentationLinkTarget::Builtin("lu"),
    },
    BuiltinDocumentationLink {
        label: "gmres",
        target: BuiltinDocumentationLinkTarget::Builtin("gmres"),
    },
    BuiltinDocumentationLink {
        label: "ichol",
        target: BuiltinDocumentationLinkTarget::Builtin("ichol"),
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ilu"),
    slug: Some("ilu"),
    summary: "Compute sparse incomplete LU factors without densifying.",
    description: "`ilu` builds managed zero-fill, Crout, or threshold-pivoted sparse factors for iterative preconditioning.",
    keywords: &["ilu", "incomplete LU", "sparse", "preconditioner"],
    related: &["lu", "gmres", "ichol"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "managed sparse ILU",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/factor/ilu.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "managed sparse ILU tests",
            location: "builtins::math::linalg::factor::ilu::tests",
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
