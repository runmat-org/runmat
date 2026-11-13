use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/ss2tf.rs") };

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ss2tf"), slug: Some("ss2tf"),
    summary: "Convert real state-space matrices to transfer-function coefficients for a selected input.",
    description: "`[b,a] = ss2tf(A,B,C,D,ni)` returns one numerator row for each output and one common denominator row. `ni` is one-based and defaults to 1. Matrix dimensions must form a valid state-space system.",
    keywords: &["ss2tf", "state space", "transfer function", "input channel"],
    related: &["ss", "tf", "tfdata"],
    sections: &[
        BuiltinDocumentationSection { heading: "Selected input", paragraphs: &[
            "For A of order n and q output rows in C, b is q-by-(n+1) and a is 1-by-(n+1). Each b row preserves the characteristic-polynomial order, including leading zero coefficients. The selected column of B and D corresponds to ni; all q outputs are returned.",
            "Inputs must be finite real Single or Double scalars or dense matrices. All-Single inputs produce Single coefficients; otherwise coefficients are Double. Explicit resident inputs are read through their owning provider and results are host values, not resident results.",
        ] },
        BuiltinDocumentationSection { heading: "Current boundaries", paragraphs: &[
            "Complex, sparse, distributed and integer matrix inputs are not admitted by this matrix form. The conversion uses characteristic and Markov polynomial operations on the host; large synchronous numerical work and a managed finite allocation envelope remain qualification obligations.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "selected-input", title: "Choose one input of a two-input system", program: "A = -2; B = [1 3]; C = [1; 2]; D = [0 1; 0 2]; [b,a] = ss2tf(A,B,C,D,2);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(b),[2 2])); assert(max(abs(b(:)-[1;2;5;10])) < 1e-10); assert(max(abs(a-[1 2])) < 1e-10);" } },
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "What does ni select?", answer: "The one-based input channel of B and D. It defaults to channel 1." }],
    links: &[SOURCE, BuiltinDocumentationLink { label: "ss", target: BuiltinDocumentationLinkTarget::Builtin("ss") }, BuiltinDocumentationLink { label: "tfdata", target: BuiltinDocumentationLinkTarget::Builtin("tfdata") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Selected input and MIMO output polynomial shape", location: "crates/runmat-runtime/src/builtins/control/model_polynomial/ss_conversion.rs" }], notes: &["Native/WASM, finite resource, and cancellation qualification remains open."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
