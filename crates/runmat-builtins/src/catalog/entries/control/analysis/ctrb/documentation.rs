use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/ctrb.rs"),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ctrb"),
    slug: Some("ctrb"),
    summary: "Form the state-space controllability matrix.",
    description: "`ctrb(A,B)` and `ctrb(sys)` assemble `[B, A*B, ..., A^(n-1)*B]` for one n-state model. The result has n rows and n*m columns when B has m inputs.",
    keywords: &["ctrb", "controllability", "reachability", "state space", "control system"],
    related: &["ss", "rank"],
    sections: &[
        BuiltinDocumentationSection { heading: "Input and output contract", paragraphs: &[
            "A must be a nonempty square host matrix and B must have the same row count and at least one column. One `ss` model supplies its stored A and B fields; model arrays, uncertain/generalized models, and sparse coefficient matrices are not yet supported.",
            "Matching real or complex single-precision matrices produce a single-precision result; matching double-precision matrices produce double. Inputs with differing precision or real/complex storage currently return a typed unsupported-input error rather than silently converting the matrices.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
            "Resident matrix inputs gather through their provider before the host calculation. Finite-memory calls currently require host-resident inputs: result storage is reserved before computation and released if the call fails or is cancelled. The operation is a fusion boundary and has no provider-side controllability kernel.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "two-state", title: "Two-state controllability", program: "A = [0 1; -2 -3]; B = [0; 1]; Co = ctrb(A,B);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(Co,[0 1;1 -3])); assert(rank(Co)==2);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does ctrb test controllability?", answer: "It forms the controllability matrix. Compare its rank with the number of states to test full controllability." },
        BuiltinDocumentationFaq { question: "Can I pass an ss model?", answer: "Yes, for one ordinary state-space model; ctrb reads only A and B." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "ss", target: BuiltinDocumentationLinkTarget::Builtin("ss") },
        BuiltinDocumentationLink { label: "MathWorks ctrb reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/control/ref/statespacemodel.ctrb.html") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Native products, shape, validation and exact result ownership", location: "crates/runmat-runtime/src/builtins/control/ctrb/tests.rs" }], notes: &["Single-model, matching native class implementation; mixed-class/model-array/provider-native forms remain open."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
