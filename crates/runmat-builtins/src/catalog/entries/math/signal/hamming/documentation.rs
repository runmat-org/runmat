use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("hamming"), slug: Some("hamming"),
    summary: "Generate Hamming windows in MATLAB and RunMat.",
    description: "`hamming(L)` returns an L-by-1 Hamming window. RunMat supports symmetric and periodic sampling, and an optional single-precision output type.",
    keywords: &["hamming", "window", "signal processing", "dsp", "fft"],
    related: &["hann", "blackman", "fft"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "`hamming(L)` returns an `L x 1` symmetric Hamming window.",
            "`hamming(L, 'periodic')` returns the periodic form used in spectral-analysis workflows.",
            "If `L` is noninteger, RunMat rounds it to the nearest integer before constructing the window.",
            "`hamming(0)` returns an empty `0 x 1` tensor.",
            "`hamming(1)` returns `1`.",
            "For the symmetric case and `L > 1`, coefficients follow `0.54 - 0.46*cos(2*pi*n/(L-1))`.",
            "The periodic form is constructed by evaluating a symmetric window of length `L + 1` and dropping the final sample."
        ] }, BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "The scalar length is host structural input. When the active provider supports the selected precision, RunMat may create the new window resident on that provider; otherwise it constructs a host window. This is automatic output placement, not an interactive gpuArray length overload.",
            "The window is not fused with adjacent operations. Symmetric and periodic forms use the same cosine-sum coefficients; periodic sampling omits the final point of a length L + 1 symmetric evaluation."
        ] }],
    examples: &[BuiltinExample { id: "example-1", title: "Generate a short Hamming window", program: "w = hamming(8);\ndisp(w')", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(w,'double')); assert(isequal(size(w),[8 1])); assert(abs(w(1)-0.08)<1e-6);" } },
        BuiltinExample { id: "example-2", title: "Generate a periodic Hamming window", program: "w = hamming(8, 'periodic');\ndisp(w')", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(w,'double')); assert(isequal(size(w),[8 1])); assert(abs(w(1)-0.08)<1e-6);" } }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "When should I use the periodic Hamming window?", answer: "Use `'periodic'` when the window will be paired with FFT-based spectral analysis. The periodic form is built from a symmetric window of length `L + 1` with the final sample removed." },
        BuiltinDocumentationFaq { question: "What shape does `hamming(L)` return?", answer: "It returns an `L x 1` column vector." },
        BuiltinDocumentationFaq { question: "Does RunMat support a `typeName` overload for `hamming`?", answer: "Yes. The runtime accepts `hamming(L, 'single')` and `hamming(L, sflag, 'single')`; the default is double. The old reference text described only the first two forms, so this page now documents the implemented forms." }],
    links: &[BuiltinDocumentationLink { label: "hann", target: BuiltinDocumentationLinkTarget::Builtin("hann") },
        BuiltinDocumentationLink { label: "blackman", target: BuiltinDocumentationLinkTarget::Builtin("blackman") },
        BuiltinDocumentationLink { label: "fft", target: BuiltinDocumentationLinkTarget::Builtin("fft") }, BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/hamming.rs") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/hamming.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Window numerical and option behavior", location: "builtins::math::signal::hamming::tests" }], notes: &["Contract maturity remains Incomplete until compiled binding, provider, native/browser and standalone-example gates run."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
