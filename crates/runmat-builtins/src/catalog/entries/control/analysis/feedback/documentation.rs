use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/feedback.rs") };
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("feedback"), slug: Some("feedback"),
    summary: "Interconnect SISO transfer functions in a feedback loop.",
    description: "`feedback(sys1,sys2)` constructs the negative-feedback closed-loop transfer function `sys1/(1+sys1*sys2)`. An optional sign of `+1` selects positive feedback, giving `sys1/(1-sys1*sys2)`. Current RunMat support is SISO `tf` models with compatible sample times and no nonzero delays.",
    keywords: &["feedback", "closed loop", "control system", "transfer function", "tf"],
    related: &["tf", "step", "isstable"],
    sections: &[
        BuiltinDocumentationSection { heading: "Forms and compatibility", paragraphs: &[
            "The MATLAB-compatible subset accepts a SISO tf object as sys1 and a compatible SISO tf object or ordinary real or complex double scalar gain as sys2; sign is -1 or +1. A scalar sys2 becomes a constant transfer function with sys1's variable and sample time. The one-argument unity-feedback form, scalar sys1, and scalar/scalar form have independent RunMat-only extension gates.",
            "Typed-integer scalar gains and signs, native-single, logical, and resident numeric arguments have separate RunMat-only gates. All eight fixed-width integer classes are admitted as scalar gains or signs; gains cross into binary64 tf coefficients, so wide integers can round. A typed-integer sign must still be exactly -1 or +1. Typed-integer tf coefficients or Ts and logical tf coefficients assigned after construction retain their model-property gates; integer model properties require exact conversion, unlike scalar gains.",
            "Both tf models must have compatible sample times. Nonzero input or output delays are not yet supported. MIMO, state-space, feed-index and signal-name topology forms remain outside the current implementation.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
            "The operation runs on the host and produces a new host tf object; it has no provider-level GPU kernel and is a fusion boundary. Valid resident scalar gains and signs gather after extension admission. Required resident coefficient properties inside valid tf objects gather through their owner during ordered model parsing; unrelated object properties are not read.",
            "The call can suspend for provider gathers and can throw for unsupported model forms, invalid sign, or failed interconnection. It does not mutate input models or use ambient filesystem, network, UI, randomness, or time services.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "negative", title: "Unity negative feedback", program: "G = tf(2, [1 3]); T = feedback(G, 1); d = T.Denominator; fprintf('d = [%g %g]\\n', d(1), d(2));", display_output: Some("d = [1 5]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(d, [1 5]));" } },
        BuiltinExample { id: "positive", title: "Positive feedback", program: "G = tf(2, [1 3]); T = feedback(G, 1, +1); d = T.Denominator; fprintf('d = [%g %g]\\n', d(1), d(2));", display_output: Some("d = [1 1]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(d, [1 1]));" } },
        BuiltinExample { id: "unity-extension", title: "RunMat unity-feedback shorthand", program: "G = tf(2, [1 3]); T = feedback(G); d = T.Denominator;", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(d, [1 5]));" } },
        BuiltinExample { id: "integer-gain", title: "RunMat integer scalar gain", program: "G = tf(2, [1 3]); T = feedback(G, uint8(1)); d = T.Denominator;", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(d, [1 5]));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does feedback support MIMO or state-space systems?", answer: "Not yet. The current implementation constructs SISO tf interconnections." },
        BuiltinDocumentationFaq { question: "Are typed-integer gains MATLAB-compatible?", answer: "Their documented parity is unverified. RunMat admits them only in RunMat compatibility mode; conversion to floating tf coefficients can round wide values." },
        BuiltinDocumentationFaq { question: "Which topology forms are missing?", answer: "Feed-index and signal-name forms require MIMO and named-I/O infrastructure that is not yet supported." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "tf", target: BuiltinDocumentationLinkTarget::Builtin("tf") },
        BuiltinDocumentationLink { label: "step", target: BuiltinDocumentationLinkTarget::Builtin("step") },
        BuiltinDocumentationLink { label: "isstable", target: BuiltinDocumentationLinkTarget::Builtin("isstable") },
        BuiltinDocumentationLink { label: "MathWorks feedback documentation", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/control/ref/inputoutputmodel.feedback.html") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Feedback signs, extension gates, integers and provider preflight", location: "crates/runmat-runtime/src/builtins/control/feedback.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM tf/feedback workflow", location: "crates/runmat-vm/tests/control.rs" },
    ], notes: &["SISO tf-only support and absent MIMO topology are explicit partial-coverage limits. Native/browser/provider qualification remains an integrated gate."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
