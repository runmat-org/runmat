use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/dlqr.rs"),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("dlqr"),
    slug: Some("dlqr"),
    summary: "Compute a discrete-time linear-quadratic state-feedback gain.",
    description: "`K = dlqr(A,B,Q,R[,N])` uses the same validated DARE solver as discrete-state-space `lqr`; `[K,S,P]` additionally returns its stabilizing Riccati solution and closed-loop poles.",
    keywords: &["dlqr", "discrete lqr", "dare", "state feedback", "control system"],
    related: &["lqr", "ss", "eig"],
    sections: &[
        BuiltinDocumentationSection { heading: "Inputs and outputs", paragraphs: &[
            "A is real n-by-n, B real n-by-m, Q symmetric n-by-n, R symmetric positive-definite m-by-m, and optional N is n-by-m. The block cost must be positive semidefinite. Values must be finite. As with the shared lqr solver, inputs normalize to host double precision and K/S are real double; P may be complex double.",
            "This is the explicit matrix-input discrete-time form. The state-space-object overload, complex weighting, descriptor dynamics and non-stabilizing solutions are not claimed. Failed convergence or an invalid DARE residual produces a typed solver error.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "The shared QR-deflated, ordered-QZ DARE solver checks the original equation residual and closed-loop stability. Resident operands gather before host computation; managed operations lacking complete output ownership reject before gathering. No provider Riccati kernel is claimed.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "scalar-stable", title: "One-state discrete regulator", program: "[K,S,P] = dlqr(1,1,1,1);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(S*S-S-1)<1e-8); assert(abs(K-S/(1+S))<1e-8); assert(abs(P)<1);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What is the default cross weight?", answer: "N defaults to the zero n-by-m matrix." },
        BuiltinDocumentationFaq { question: "How does dlqr differ from lqr(A,B,Q,R)?", answer: "Matrix-input lqr uses the continuous CARE; dlqr uses the discrete DARE for the same matrix inputs." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "lqr", target: BuiltinDocumentationLinkTarget::Builtin("lqr") },
        BuiltinDocumentationLink { label: "MathWorks dlqr reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/control/ref/lti.dlqr.html") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar DARE residual, gain and stable pole", location: "crates/runmat-runtime/src/builtins/control/dlqr.rs" }], notes: &["Standard real explicit-matrix DARE subset; shared lqr numerical owner, no complex/generalized/factor claims."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
