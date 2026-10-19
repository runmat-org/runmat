use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Syntax", paragraphs: &[
        "T = schur(A); [U,T] = schur(A); either form accepts an optional 'real' or 'complex' mode.",
        "For real A, default real mode returns a real quasitriangular T. Complex mode returns triangular complex T. Complex A always uses complex form.",
    ] },
    BuiltinDocumentationSection { heading: "Behavior and placement", paragraphs: &[
        "A must be a finite dense square single or double matrix. U and T preserve its floating class; A = U*T*U' with orthogonal or unitary U.",
        "The portable host implementation uses a bounded nalgebra Schur iteration. Resident GPU inputs are not admitted because no Schur provider hook or transfer policy is declared.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "schur-rotation", title: "Real Schur blocks retain a complex pair",
    program: "A = [0 -1; 1 0]; [U,T] = schur(A); e = ordeig(T);",
    display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions { source: "assert(norm(A-U*T*U') < 1e-10); assert(norm(U'*U-eye(2)) < 1e-10); assert(norm(sort(e)-sort([1i;-1i])) < 1e-10);" },
}];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("schur"), slug: Some("schur"),
    summary: "Compute an ordinary real or complex Schur factorization.",
    description: "Compute a retained real quasitriangular or complex triangular factorization of a dense square matrix.",
    keywords: &["schur", "eigenvalues", "factorization"], related: &["ordeig", "ordschur", "rsf2csf", "eig"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: &[], links: &[], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink {
        label: "Schur runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/factor/schur/mod.rs"),
    }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest,
        label: "Schur factor invariants", location: "builtins::math::linalg::factor::schur::tests" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
