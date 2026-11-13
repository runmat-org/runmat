use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/bode.rs") };

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("bode"), slug: Some("bode"),
    summary: "Compute or plot SISO transfer-function magnitude and phase.",
    description: "`[mag,phase,wout] = bode(sys,w)` evaluates a scalar, zero-delay `tf` model. Discrete models require a known positive sample period. Magnitude is absolute (not dB); unwrapped phase is in degrees. Both response arrays have size 1-by-1-by-N and `wout` is a column in rad/s. Statement form draws magnitude in dB and phase on logarithmic frequency axes when plotting is available. MIMO/state-space/model-array, delay, multiple-system and LineSpec forms are not yet supported.",
    keywords: &["bode", "frequency response", "magnitude", "phase", "control"],
    related: &["tf", "freqresp", "nyquist", "bandwidth"],
    sections: &[
        BuiltinDocumentationSection { heading: "Frequency and phase", paragraphs: &["Continuous models are evaluated on `s=1i*w`; discrete models on `z=exp(1i*w*Ts)`. An omitted frequency vector uses a deterministic grid derived from poles and zeros. Phase is unwrapped in source frequency order; supply an ordered grid for the usual continuous plot."] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &["The shared host transfer-function evaluator computes the numeric response. Context-bound finite execution without a complete output owner rejects before model-property gathering. Statement-form plotting may suspend and uses the existing plotting service; numeric outputs have no UI effect."] },
    ],
    examples: &[BuiltinExample { id: "first_order", title: "Magnitude and phase at one radian per second", program: "H = tf(1,[1 1]); [mag,phase,w] = bode(H,1);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(mag(1,1,1)-1/sqrt(2)) < 1e-10); assert(abs(phase(1,1,1)+45) < 1e-10); assert(w(1)==1);" } }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Are returned magnitudes in decibels?", answer: "No. Numeric `mag` is absolute; only the statement-form plot converts magnitude to dB." }],
    links: &[BuiltinDocumentationLink { label: "tf", target: BuiltinDocumentationLinkTarget::Builtin("tf") }, SOURCE],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[], notes: &["The new leaf awaits catalog composition and integrated product qualification. SISO tf-only behavior is partial coverage."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
