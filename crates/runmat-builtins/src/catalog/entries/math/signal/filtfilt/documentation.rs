use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("filtfilt"),
    slug: Some("filtfilt"),
    summary: "Filter a signal forward and backward to remove phase delay.",
    description: "`filtfilt(b,a,x)` applies the digital filter specified by numerator `b` and denominator `a` in both directions. The implemented form accepts a vector and preserves its row or column orientation. The leading denominator coefficient is normalized before filtering.",
    keywords: &["filtfilt", "filter", "zero phase", "FIR", "IIR", "GPU"],
    related: &["filter", "fir1", "butter"],
    sections: &[
        BuiltinDocumentationSection { heading: "Supported form", paragraphs: &[
            "This implementation accepts vector signals, not matrix or explicit-dimension forms. Coefficients must be nonempty vectors and `a(1)` must be nonzero.",
            "The three-argument path initializes from the steady-state response and extends both endpoints by odd reflection (linear endpoint continuation) with `3*(max(numel(a),numel(b))-1)` samples. The input length must exceed that reflection length except for zero-state filters.",
            "Forward and reverse passes cancel phase delay. A single-precision input in any of the three roles yields single output; double-only and RunMat integer/logical forms yield double output. Integer/logical input is a RunMat extension, not a claim of MATLAB parity.",
        ] },
        BuiltinDocumentationSection { heading: "Placement and ownership", paragraphs: &[
            "An admitted real floating resident signal uses one provider-owned zero-phase route for endpoint reflection, both filter passes, reversal, and trimming. Intermediate signal and pass values remain resident; the route reads back scalar finite predicates and uploads its prepared coefficients and initial state. Complex signals use the host reference calculation.",
            "When that complete resident route is unavailable, a compatible real resident input can use host-staged provider passes: the signal is gathered for endpoint reflection, each prepared forward or reverse pass runs through `iir_filter`, and pass results are read back for reversal and trimming before resident publication. This route includes transfers and is not an entirely device-resident calculation.",
            "If whole-route capability is unavailable before provider effects, RunMat can choose the host reference route. Operational failures after route submission preserve their cause and are not replayed as host filtering.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "causal-comparison", title: "Compare causal and zero-phase filtering", program: "b = ones(1,3)/3; a = 1; x = 1:20; y_causal = filter(b,a,x); y_zero = filtfilt(b,a,x); fprintf('%.4f %.4f\\n', y_causal(8), y_zero(8));", display_output: Some("7.0000 8.0000"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(y_causal(8)-7)<1e-12); assert(abs(y_zero(8)-8)<1e-12);" } },
        BuiltinExample { id: "fir1", title: "Use coefficients from fir1", program: "b = fir1(10,0.25,'low'); x = 1:40; y = filtfilt(b,1,x);", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(numel(y)==40); assert(all(isfinite(y)));" } },
        BuiltinExample { id: "gpu", title: "Use a resident input", program: "g = gpuArray(1:40); b = ones(1,3)/3; y_gpu = filtfilt(b,1,g); y = gather(y_gpu);", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(numel(y)==40); assert(max(abs(y-filtfilt(b,1,1:40)))<1e-10);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "How does `filtfilt` differ from `filter`?", answer: "`filter` is causal; `filtfilt` applies the filter in both directions to cancel phase delay while squaring its magnitude response." },
        BuiltinDocumentationFaq { question: "Does `filtfilt` support matrices?", answer: "The currently implemented public form accepts vectors. Matrix and explicit-dimension forms are not yet supported." },
        BuiltinDocumentationFaq { question: "Can I use Butterworth or FIR coefficients?", answer: "Yes. Coefficients from `butter`, `fir1`, or a hand-written digital filter are accepted when `a(1)` is nonzero and the vector meets the reflection-length requirement." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "filter", target: BuiltinDocumentationLinkTarget::Builtin("filter") },
        BuiltinDocumentationLink { label: "fir1", target: BuiltinDocumentationLinkTarget::Builtin("fir1") },
        BuiltinDocumentationLink { label: "butter", target: BuiltinDocumentationLinkTarget::Builtin("butter") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/filtfilt.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/filtfilt.rs") },
            BuiltinDocumentationLink { label: "Resident zero-phase route", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/filtfilt/resident_zero_phase.rs") },
            BuiltinDocumentationLink { label: "Host-staged provider passes", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/filtfilt/provider_pass.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Host numerical behavior", location: "builtins::math::signal::filtfilt::tests::host" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Provider routes", location: "builtins::math::signal::filtfilt::tests::provider_routes" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Provider output admission", location: "builtins::math::signal::filtfilt::tests::provider_output" },
        ],
        notes: &["The resident example requires an available WGPU device. Provider precision and supported filter forms determine which route can execute; the host-staged route includes signal transfers."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
