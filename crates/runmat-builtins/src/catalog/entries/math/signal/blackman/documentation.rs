use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("blackman"), slug: Some("blackman"),
    summary: "Generate a Blackman window vector in MATLAB and RunMat.",
    description: "`blackman` generates a Blackman window from a nonnegative scalar length. The length accepts single, double, or any of the eight built-in integer classes; floating lengths round to the nearest integer.",
    keywords: &["blackman", "window", "signal processing", "dsp", "fft"],
    related: &["hann", "hamming", "fft"],
    sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "`blackman(L)` returns an `L x 1` symmetric Blackman window.",
            "`blackman(L, 'periodic')` returns the periodic form used in spectral-analysis workflows.",
            "`blackman(..., 'single')` returns a single-precision window; `'double'` is the default.",
            "Length `L` accepts single, double, int8, int16, int32, int64, uint8, uint16, uint32, or uint64; noninteger floating values round to the nearest integer.",
            "A logical scalar length is available only as an explicitly gated RunMat extension.",
            "`blackman(0)` returns an empty `0 x 1` tensor.",
            "`blackman(1)` returns `1`.",
            "For the symmetric case and `L > 1`, coefficients use `cospi` evaluation of `0.42 - 0.5*cospi(2*n/(L-1)) + 0.08*cospi(4*n/(L-1))`, matching the current enhanced-accuracy algorithm.",
            "The periodic form is constructed by evaluating a symmetric window of length `L + 1` and dropping the final sample."
        ] }, BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "The scalar length is host structural input. When the active provider supports the selected precision, RunMat may create the new window resident on that provider; otherwise it constructs a host window. This is automatic output placement, not an interactive gpuArray length overload.",
            "The window is not fused with adjacent operations. Symmetric and periodic forms use the same cosine-sum coefficients; periodic sampling omits the final point of a length L + 1 symmetric evaluation."
        ] }],
    examples: &[BuiltinExample { id: "example-1", title: "Generate a short Blackman window", program: "w = blackman(8);\ndisp(w')", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(w,'double')); assert(isequal(size(w),[8 1])); assert(abs(w(1)-0)<1e-6);" } },
        BuiltinExample { id: "example-2", title: "Generate a periodic Blackman window", program: "w = blackman(8, 'periodic');\ndisp(w')", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(w,'double')); assert(isequal(size(w),[8 1])); assert(abs(w(1)-0)<1e-6);" } },
        BuiltinExample { id: "example-3", title: "Generate a single-precision Blackman window", program: "w = blackman(8, 'single');\nclass(w)", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(w,'single')); assert(isequal(size(w),[8 1]));" } }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Why would I use `blackman` instead of `hann` or `hamming`?", answer: "A Blackman window is often used when you want lower spectral leakage at the cost of a wider main lobe. It is a common tradeoff choice in spectral analysis workflows." },
        BuiltinDocumentationFaq { question: "What is the difference between the symmetric and periodic Blackman window?", answer: "Use the symmetric form for filter-design style workflows. Use `'periodic'` for FFT-based spectral analysis, where RunMat constructs a symmetric window of length `L + 1` and removes the final sample." },
        BuiltinDocumentationFaq { question: "Does `blackman` support single precision?", answer: "Yes. Pass `'single'` as the final option to request a single-precision output. The default output type is double precision." },
        BuiltinDocumentationFaq { question: "Does an integer length produce an integer window?", answer: "No. Every integer length class is a structural input; output is double by default or single when requested explicitly." },
        BuiltinDocumentationFaq { question: "Does `blackman` accept a gpuArray input?", answer: "No interactive gpuArray input form exists because the only numeric input is a host scalar length. GPU code generation is a separate documented capability; RunMat may internally materialize the new floating window on its active provider." }],
    links: &[BuiltinDocumentationLink { label: "hann", target: BuiltinDocumentationLinkTarget::Builtin("hann") },
        BuiltinDocumentationLink { label: "hamming", target: BuiltinDocumentationLinkTarget::Builtin("hamming") },
        BuiltinDocumentationLink { label: "fft", target: BuiltinDocumentationLinkTarget::Builtin("fft") }, BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/blackman.rs") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/blackman.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Window numerical and option behavior", location: "builtins::math::signal::blackman::tests" }], notes: &["Contract maturity remains Incomplete until compiled binding, provider, native/browser and standalone-example gates run."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
