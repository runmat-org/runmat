use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("hann"), slug: Some("hann"),
    summary: "Generate Hann windows in MATLAB and RunMat.",
    description: "`hann(L)` returns an L-by-1 Hann window using the compatibility target's `cospi` formulation. `L` accepts floating input and every integer class; logical scalar lengths are available only when RunMat extensions are enabled. Symmetric/periodic forms and optional single-precision output follow MATLAB semantics.",
    keywords: &["hann", "window", "signal processing", "dsp", "fft"],
    related: &["hamming", "blackman", "fft"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "`hann(L)` returns an `L x 1` symmetric Hann window.",
            "`hann(L, 'periodic')` returns the periodic form used in spectral-analysis workflows.",
            "`hann(..., 'single')` returns a single-precision window; `'double'` is the default.",
            "If `L` is noninteger, RunMat rounds it to the nearest integer before constructing the window.",
            "`hann(0)` returns an empty `0 x 1` tensor.",
            "`hann(1)` returns `1`.",
            "For the symmetric case and `L > 1`, coefficients use the compatibility target's `cospi` form `0.5 - 0.5*cospi(2*n/(L-1))`.",
            "The periodic form is constructed by evaluating a symmetric window of length `L + 1` and dropping the final sample."
        ] }, BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "The scalar length is host structural input. When the active provider supports the selected precision, RunMat may create the new window resident on that provider; otherwise it constructs a host window. This is automatic output placement, not an interactive gpuArray length overload.",
            "The window is not fused with adjacent operations. Symmetric and periodic forms use the same cosine-sum coefficients; periodic sampling omits the final point of a length L + 1 symmetric evaluation."
        ] }],
    examples: &[BuiltinExample { id: "example-1", title: "Generate a short Hann window", program: "w = hann(8);\ndisp(w')", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(w,'double')); assert(isequal(size(w),[8 1])); assert(abs(w(1)-0)<1e-6);" } },
        BuiltinExample { id: "example-2", title: "Generate a periodic Hann window for spectral analysis", program: "w = hann(8, 'periodic');\ndisp(w')", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(w,'double')); assert(isequal(size(w),[8 1])); assert(abs(w(1)-0)<1e-6);" } },
        BuiltinExample { id: "example-3", title: "Generate a single-precision Hann window", program: "w = hann(8, 'single');\nclass(w)", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(w,'single')); assert(isequal(size(w),[8 1]));" } }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "What is the difference between the symmetric and periodic Hann window?", answer: "Use the default symmetric window for filter-design style workflows. Use `'periodic'` for spectral analysis, where MATLAB and RunMat construct a symmetric window of length `L + 1` and drop the last sample so the sequence matches the periodicity assumptions of the DFT." },
        BuiltinDocumentationFaq { question: "What shape does `hann(L)` return?", answer: "It returns an `L x 1` column vector, matching MATLAB's window-function convention." },
        BuiltinDocumentationFaq { question: "Does `hann` support single precision?", answer: "Yes. Pass `'single'` as the final option to request a single-precision output. The default output type is double precision." }],
    links: &[BuiltinDocumentationLink { label: "hamming", target: BuiltinDocumentationLinkTarget::Builtin("hamming") },
        BuiltinDocumentationLink { label: "blackman", target: BuiltinDocumentationLinkTarget::Builtin("blackman") },
        BuiltinDocumentationLink { label: "fft", target: BuiltinDocumentationLinkTarget::Builtin("fft") }, BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/hann.rs") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/hann.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Window numerical and option behavior", location: "builtins::math::signal::hann::tests" }], notes: &["Contract maturity remains Incomplete until compiled binding, provider, native/browser and standalone-example gates run."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
