use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/care.rs"),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("care"),
    slug: Some("care"),
    summary: "Solve a standard continuous-time algebraic Riccati equation.",
    description: "`X = care(A,B,Q[,R[,S]])` solves `A'*X + X*A - (X*B+S)*inv(R)*(B'*X+S') + Q = 0`. Optional outputs `[X,L,G]` provide the solution, closed-loop eigenvalues and feedback gain.",
    keywords: &["care", "riccati", "continuous time", "lqr", "control system"],
    related: &["lqr", "ss", "eig"],
    sections: &[
        BuiltinDocumentationSection { heading: "Supported standard form", paragraphs: &[
            "A is nonempty real n-by-n, B real n-by-m, Q symmetric n-by-n, R symmetric nonsingular m-by-m, and S is optional n-by-m. R defaults to identity and S to zero. Inputs must be finite; calculation and outputs use host double precision. Indefinite Q or R are accepted when the equation has a finite stabilizing solution; the nonnegative-cost restrictions of lqr and dlqr do not apply to care.",
            "The descriptor-pair E argument, factorized output form, fourth diagnosis report, complex data, and singular R are not yet supported. A missing stabilizing solution returns a typed solver error; the empty-result and report conventions of MATLAB are not yet implemented.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "The standard CARE uses the shared QR-deflated, ordered-QZ Riccati solver and checks its original-equation residual and closed-loop stability. Valid resident operands gather before host computation; managed execution without a complete result owner rejects before gathering.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "scalar-stable", title: "Scalar stabilizing CARE", program: "[X,L,G] = care(-1,1,1,1);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(-2*X-X*X+1)<1e-8); assert(abs(G-X)<1e-8); assert(real(L)<0);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What is the output order?", answer: "care returns the Riccati solution X, closed-loop eigenvalues L, then feedback gain G. This differs from lqr's K,S,e order." },
        BuiltinDocumentationFaq { question: "Does care support generalized descriptor E?", answer: "Not yet. The current implementation is the standard E=I continuous-time equation only." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "lqr", target: BuiltinDocumentationLinkTarget::Builtin("lqr") },
        BuiltinDocumentationLink { label: "MathWorks care reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/control/ref/care.html") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar CARE residual, indefinite weights, gain and stable pole", location: "crates/runmat-runtime/src/builtins/control/care.rs" }], notes: &["Standard real CARE uses the shared ordered-QZ solver; generalized, factor and report forms are outstanding."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
