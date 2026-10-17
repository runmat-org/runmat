use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/tfdata.rs") };

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("tfdata"), slug: Some("tfdata"),
    summary: "Extract transfer-function numerator, denominator and sample time from a control model.",
    description: "`[num,den,Ts] = tfdata(sys)` returns coefficient cells and sample time. A state-space MIMO model yields an output-by-input coefficient cell grid. A homogeneous model array adds its model dimensions after the output and input dimensions; scalar indices select one array entry. For a scalar SISO model, `'v'` returns coefficient row vectors instead of cells.",
    keywords: &["tfdata", "transfer function", "coefficients", "state space"],
    related: &["tf", "ss", "ss2tf"],
    sections: &[
        BuiltinDocumentationSection { heading: "Results", paragraphs: &[
            "Scalar tf or ss models and nonempty homogeneous arrays of those models are accepted. Without the vector selector, numerator and denominator are output-by-input cells containing coefficient row vectors (1-by-1 for scalar SISO). An array returns cell pages in column-major model order. The optional third output is the common sample time; fourth and fifth uncertainty outputs are empty for deterministic models.",
            "Positive scalar indices in `tfdata(sys,J1,...,JN)` select one model-array entry; one index uses column-major linear order. The vector selector `tfdata(sys,'v')` applies only to a scalar SISO model. State-space conversion uses the shared characteristic/Markov polynomial owner; empty arrays, zpk, and uncertain models remain unsupported.",
        ] },
        BuiltinDocumentationSection { heading: "Placement", paragraphs: &["Accepted resident model properties are gathered through their exact provider owner. The output is host-side coefficient data; no GPU coefficient kernel is claimed."] },
    ],
    examples: &[
        BuiltinExample { id: "coefficient-vectors", title: "Read SISO coefficient vectors", program: "G = tf([2 4],[1 3]); [n,d] = tfdata(G,'v');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(n,[2 4])); assert(isequal(d,[1 3]));" } },
        BuiltinExample { id: "mimo-state-space", title: "Read a MIMO coefficient grid", program: "G = ss(-2,[1 3],[1;2],[0 1;0 2]); [n,d,ts] = tfdata(G);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(n),[2 2])); assert(isequal(n{1,2},[1 5])); assert(isequal(n{2,2},[2 10])); assert(isequal(d{1,2},[1 2])); assert(ts == 0);" } },
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "When are cells returned?", answer: "Without the SISO `'v'` selector, coefficient outputs are cell arrays." }],
    links: &[SOURCE, BuiltinDocumentationLink { label: "tf", target: BuiltinDocumentationLinkTarget::Builtin("tf") }, BuiltinDocumentationLink { label: "ss2tf", target: BuiltinDocumentationLinkTarget::Builtin("ss2tf") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Coefficient extraction, vector selector, and model-array indexing", location: "crates/runmat-runtime/src/builtins/control/tfdata.rs" }], notes: &["Empty model arrays, resource and cancellation bounds remain open."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
