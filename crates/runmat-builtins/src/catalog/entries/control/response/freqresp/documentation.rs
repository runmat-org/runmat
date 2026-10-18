use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/freqresp.rs") };

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("freqresp"), slug: Some("freqresp"),
    summary: "Evaluate a SISO transfer function on an angular-frequency grid.",
    description: "`freqresp(sys,w)` evaluates a scalar, zero-delay `tf` model at nonnegative host frequencies in rad/s. Discrete models require a known positive sample period; an unspecified `Ts` cannot map physical frequency to a z-plane point. The complex double response has size 1-by-1-by-N; `[H,wout]` also returns the frequency column. Without `w`, a deterministic model-derived grid is selected. MIMO, state-space, model arrays, delays and frequency-unit selectors are not yet supported.",
    keywords: &["freqresp", "frequency response", "transfer function", "control"],
    related: &["tf", "evalfr", "bode", "nyquist"],
    sections: &[
        BuiltinDocumentationSection { heading: "Frequency points", paragraphs: &["Continuous models use `s = 1i*w`; discrete models use `z = exp(1i*w*Ts)`. Supplied frequencies must be a nonempty real host scalar or vector, finite and nonnegative. An exact pole or unrepresentable response is an error."] },
        BuiltinDocumentationSection { heading: "Placement", paragraphs: &["Host evaluation borrows the canonical transfer-function polynomial evaluator. A finite managed operation without a complete output owner rejects before gathering model properties. Direct resident frequency vectors are not yet supported."] },
    ],
    examples: &[BuiltinExample { id: "two_frequencies", title: "Evaluate at zero and one radian per second", program: "H = tf(1,[1 1]); [fr,w] = freqresp(H,[0 1]);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(fr(1,1,1)-1) < 1e-10); assert(abs(fr(1,1,2)-(0.5-0.5i)) < 1e-10); assert(isequal(size(fr),[1 1 2]));" } }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Are frequencies measured in hertz?", answer: "No. The current form uses angular frequency in radians per time unit." }],
    links: &[BuiltinDocumentationLink { label: "tf", target: BuiltinDocumentationLinkTarget::Builtin("tf") }, SOURCE],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[], notes: &["The unlinked leaf awaits catalog composition and integrated product tests. SISO tf-only support is partial coverage."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
