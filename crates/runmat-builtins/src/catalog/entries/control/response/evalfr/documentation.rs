use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/evalfr.rs") };

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("evalfr"), slug: Some("evalfr"),
    summary: "Evaluate a SISO transfer function at one complex s- or z-plane point.",
    description: "`evalfr(sys,x)` evaluates a scalar, zero-delay `tf` model at `x` in the continuous s plane or discrete z plane. A discrete model with unspecified sample period can still be evaluated at an explicit z point. Forward and inverse discrete coefficient conventions are honored. The current implementation accepts a finite host scalar point and returns a real or complex double scalar. State-space, model arrays, delays and generalized models remain unsupported.",
    keywords: &["evalfr", "frequency response", "transfer function", "control"],
    related: &["tf", "freqresp", "bode"],
    sections: &[
        BuiltinDocumentationSection { heading: "Point semantics", paragraphs: &["For continuous-time frequency response at angular frequency `w`, pass `1i*w`; for discrete time pass `exp(1i*w*Ts)`. `evalfr` does not interpret `x` as a frequency by itself. Exact poles at `x` and unrepresentable finite response are errors."] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &["The canonical transfer-function polynomial evaluator runs on the host. A context-bound finite operation without a complete output owner rejects before gathering model properties; the unbounded path may gather resident model properties through their provider."] },
    ],
    examples: &[BuiltinExample { id: "continuous_point", title: "Evaluate a continuous transfer function", program: "H = tf(2,[1 3]); y = evalfr(H,1i);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(y - (0.6-0.2i)) < 1e-10);" } }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Does evalfr accept a frequency in rad/s directly?", answer: "No. Supply the matching s- or z-plane point; use freqresp for a frequency vector." }],
    links: &[BuiltinDocumentationLink { label: "tf", target: BuiltinDocumentationLinkTarget::Builtin("tf") }, SOURCE],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[], notes: &["New leaf implementation awaits catalog and runtime composition plus product qualification. The SISO tf-only domain is partial coverage."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
