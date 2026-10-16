use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Generalized Schur form",
        paragraphs: &[
            "[AA,BB,Q,Z] = qz(A,B) factors a finite dense square floating pencil without forming inv(B) or B\\A. The retained transforms satisfy Q*A*Z=AA and Q*B*Z=BB. Singular B is admitted.",
            "The documented default is complex triangular form. qz(A,B,'real') requires real coefficients and retains real 1-by-1 and conjugate-pair 2-by-2 blocks. Single/Double class follows the wider coefficient class; all outputs are host-resident in this implementation.",
            "The optional fifth and sixth outputs are right and left generalized eigenvectors derived from the retained Schur factors, not a second full QZ pass. A regular defective pencil may have dependent eigenvector columns. A 0/0 indeterminate eigenvalue or an uncertifiable numerical solve raises a typed numerical error rather than publishing a fabricated vector.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Current resource boundary",
        paragraphs: &[
            "This portable sequential backend has no provider QZ hook. Resident inputs are rejected before factorization; no implicit provider gather or false GPU output is claimed. Long synchronous QZ iteration does not yet expose a cooperative CPU-budget checkpoint, and all four Schur factors are currently retained even for a one-output request.",
            "The faer 0.24.4 low-level QZ kernel does not return a convergence status. RunMat checks factor finiteness and upper/quasitriangular structure before publication, but full execution and native/WASM qualification remain pending.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "qz-singular-pencil",
        title: "Factor a pencil with singular B",
        program: "A = [2 0; 0 3]; B = [1 0; 0 0]; [AA,BB,Q,Z] = qz(A,B);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(norm(Q*A*Z-AA)<1e-8); assert(norm(Q*B*Z-BB)<1e-8);",
        },
    },
    BuiltinExample {
        id: "qz-real-conjugate-pair",
        title: "Keep a real conjugate block",
        program: "A = [0 -1; 1 0]; B = eye(2); [AA,BB,Q,Z] = qz(A,B,'real');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isreal(AA)); assert(norm(Q*A*Z-AA)<1e-8); assert(norm(Q*B*Z-BB)<1e-8);",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("qz"),
    slug: Some("qz"),
    summary: "Factor a dense generalized matrix pencil without inverting B.",
    description: "Portable real or complex generalized Schur factorization with retained transforms and optional eigenvectors.",
    keywords: &["qz", "generalized Schur", "matrix pencil"],
    related: &["eig", "ordeig", "ordqz", "schur"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Retained generalized Schur implementation",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/factor/qz/algorithm.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Singular pencil and reconstruction source fixture",
            location: "builtins::math::linalg::factor::qz::tests::complex_qz_reconstructs_singular_b_without_inverse",
        }],
        notes: &["Source-only candidate: build, numerical examples, native/WASM execution, and resource/cancellation gates remain open."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
