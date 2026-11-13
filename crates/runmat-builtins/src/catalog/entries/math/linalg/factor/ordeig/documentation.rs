use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[BuiltinDocumentationSection { heading: "Ordered Schur spectrum", paragraphs: &[
    "E = ordeig(T) reads 1-by-1 and 2-by-2 diagonal blocks of an ordinary Schur form in their existing order. A nonzero real subdiagonal marks an indivisible 2-by-2 block, even when it is tiny.",
    "E = ordeig(AA,BB) reads the retained generalized Schur spectrum in block order. Singular BB can yield infinite eigenvalues; 0/0 homogeneous pairs yield indeterminate NaN. Real 2-by-2 blocks are kept together, and the pair is never reduced through B\\A.",
] }];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "ordeig-ordered",
        title: "Read a Schur spectrum without sorting",
        program: "T = diag([3 -1 2]); e = ordeig(T);",
        display_output: Some("e = [3; -1; 2]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(e,[3;-1;2]));",
        },
    },
    BuiltinExample {
        id: "ordeig-generalized-infinite",
        title: "Read a generalized Schur spectrum with singular BB",
        program: "A = [2 0; 0 3]; B = [1 0; 0 0]; [AA,BB] = qz(A,B); e = ordeig(AA,BB);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(e(1)-2)<1e-8 || abs(e(2)-2)<1e-8); assert(any(isinf(e)));",
        },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("ordeig"), slug: Some("ordeig"),
    summary: "Extract ordinary or generalized Schur eigenvalues in factor order.",
    description: "Read ordinary or generalized Schur eigenvalues in retained factor order without reordering or inverting the second factor.",
    keywords: &["ordeig", "schur", "eigenvalues"], related: &["schur", "ordschur", "qz"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: &[], links: &[], media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Ordinary Schur block spectrum",
                target: BuiltinDocumentationLinkTarget::Source(
                    "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/factor/schur/form.rs",
                ),
            },
            BuiltinDocumentationLink {
                label: "Generalized Schur block spectrum",
                target: BuiltinDocumentationLinkTarget::Source(
                    "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/factor/qz/eigvals.rs",
                ),
            },
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "tiny ordinary complex-pair block",
                location: "builtins::math::linalg::factor::schur::tests::tiny_real_schur_subdiagonal_is_structural_not_discarded",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "generalized real pair and infinite value",
                location: "builtins::math::linalg::factor::qz::tests::ordeig_retains_real_conjugate_block_order_and_infinity",
            },
        ],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
