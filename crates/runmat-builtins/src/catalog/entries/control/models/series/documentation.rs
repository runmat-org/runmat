use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/series.rs") };
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("series"), slug: Some("series"),
    summary: "Connect two supported dynamic-system models in series.",
    description: "`series(sys1,sys2)` connects sys1 upstream of sys2. For transfer functions this is sys2*sys1; for state-space models the result retains the state-space representation and source-ordered dynamics.",
    keywords: &["series", "cascade", "state space", "transfer function"],
    related: &["tf", "ss", "feedback"],
    sections: &[
        BuiltinDocumentationSection { heading: "Models and channels", paragraphs: &[
            "`series(sys1,sys2,out1,in2)` connects selected upstream outputs to selected downstream inputs. The distinct positive index vectors must have equal lengths; unconnected upstream outputs and downstream inputs are dropped. Direct tf-by-tf SISO composition retains a tf model; a composition involving an ss model produces an ss model.",
            "`series(sys1,sys2,'name')` connects uniquely matching nonempty upstream OutputName and downstream InputName entries; it rejects duplicate or absent matches. Homogeneous nonempty tf/ss model arrays of equal shape connect corresponding entries; a scalar supported model may connect to each array entry. A real scalar gain scales every channel of a MIMO ss model. Sample times must match and nonzero delays are unsupported. Mixed tf/ss composition needs a proper real tf realization; complex or improper mixed models are currently unsupported. A scalar complex gain is supported for tf-by-tf SISO composition. Typed-integer scalar gains are a RunMat extension and round at the same binary64 model-arithmetic boundary as tf multiplication and feedback.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "State-space composition is host-side matrix algebra. Required resident model properties are downloaded through their exact owners after validation; no GPU series kernel or fusion route is claimed. The call may suspend and may throw on invalid models or provider failures.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "tf-cascade", title: "Cascade transfer functions", program: "g1 = tf(2,[1 1]); g2 = tf([1 2],[1 3]); h = series(g1,g2);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(h.Numerator,[2 4])); assert(isequal(h.Denominator,[1 4 3]));" } },
        BuiltinExample { id: "ss-cascade", title: "Cascade state-space systems", program: "g1 = ss(-1,1,1,0); g2 = ss(-2,1,1,0); h = series(g1,g2);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(h),'ss')); assert(isequal(size(h.A),[2 2])); assert(abs(h.D) < 1e-12);" } },
        BuiltinExample { id: "selected-mimo", title: "Connect a selected MIMO channel", program: "g1 = ss(-1,[1 2],[3;4],[0 1;2 0]); g2 = ss(-2,[5 6],7,[8 9]); h = series(g1,g2,2,1);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(h.B),[2 2])); assert(isequal(size(h.D),[1 2])); assert(isequal(h.D,[16 0]));" } },
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Which model is upstream?", answer: "sys1 is upstream. For SISO transfer functions, series(sys1,sys2) equals sys2*sys1." }],
    links: &[SOURCE, BuiltinDocumentationLink { label: "tf", target: BuiltinDocumentationLinkTarget::Builtin("tf") }, BuiltinDocumentationLink { label: "ss", target: BuiltinDocumentationLinkTarget::Builtin("ss") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "TF and state-space cascade and channel dimensions", location: "crates/runmat-runtime/src/builtins/control/model_polynomial/series.rs" }], notes: &["Empty model arrays, other model families, improper or complex mixed realizations, browser/native integration and quota/cooperation qualification remain open."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
