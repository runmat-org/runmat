use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,

    title: Some("periodogram"),
    slug: Some("periodogram"),
    summary: "Estimate power spectral density using a periodogram.",
    description: "`periodogram(x)` estimates the power spectral density of a signal by windowing the full input and evaluating its DFT. Optional arguments select the analysis window, DFT length or explicit frequency grid, sampling frequency, frequency range, and spectrum scale.",

    keywords: &["periodogram", "PSD", "power spectrum", "spectral density", "signal processing", "FFT"],
    related: &["pwelch", "spectrogram", "fft", "hamming"],

    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["RunMat returns the computed Pxx and F outputs in the signal's floating precision: single signals produce single PSD and frequency arrays, and double signals produce double arrays. Window and sampling-control classes do not change this rule. This is RunMat's policy, not a claim of verified MATLAB class parity. The host reference computes FFT and window arithmetic in double and casts once when publishing single outputs.",
                "`pxx = periodogram(x)` returns a one-sided PSD estimate for real inputs and a two-sided estimate for complex inputs.",
                "Matrix inputs are processed column-wise; `pxx` has one output column per input column.",
                "`[pxx, f] = periodogram(...)` also returns the frequency grid. Without `fs`, frequencies are in radians/sample; with `fs`, frequencies are in Hz. Passing `fs` as `[]` uses 1 Hz units.",
                "`window` must be a finite real vector matching the signal length. `[]` uses a rectangular window.",
                "`nfft` defaults to `max(256, 2^nextpow2(length(x)))` and controls the DFT grid. If `nfft` is shorter than the window, windowed samples are summed modulo `nfft` before the DFT. A real frequency vector may be supplied in the same position to evaluate explicit frequencies.",
                "The optional range selector accepts `'onesided'`, `'twosided'`, or `'centered'`; `'onesided'` is valid only for real-valued signals.",
                "The optional spectrum type accepts `'psd'` (default) or `'power'`; `'power'` reports power spectrum values using coherent-gain normalization.",
                "Typed-integer signals and typed-integer numeric controls are separate RunMat extensions. Structural DFT lengths are decoded exactly; signals, windows, frequency grids, and sample rates must be exactly representable before entering the floating spectral domain."] },
        BuiltinDocumentationSection { heading: "Execution and ownership", paragraphs: &["Resident routes use the signal's exact owner and native class. Typed unsupported capabilities select the reference path; operational failures retain their source and do not replay execution.",
                "Requested computed outputs use checked class-preserving restoration after reference evaluation. Explicit placement requires successful restoration; automatic placement may return a host output when restoration is unsupported. Original inputs and earlier outputs remain protected.",
                "Uniform-grid periodograms keep GPU-resident signal data on device through windowing, folded framing when needed, FFT, frequency selection, and power scaling. Explicit-frequency forms fall back to host execution."] }],

    examples: &[
        BuiltinExample { id: "example-1", title: "PSD of a sine wave", program: "fs = 1000;\nt = (0:999) / fs;\nx = sin(2*pi*100*t);\n[pxx, f] = periodogram(x, [], 1024, fs);", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(pxx,'double')); assert(size(pxx,1)==numel(f));" } },
        BuiltinExample { id: "example-2", title: "Use a Hamming window and power spectrum scaling", program: "x = 1.8*cos(2*pi*(0:127)/16);\n[pxx, f] = periodogram(x, hamming(128), 128, 'power');", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(pxx,'double')); assert(size(pxx,1)==numel(f));" } },
        BuiltinExample { id: "example-3", title: "Two-sided complex spectrum", program: "x = exp(1i*2*pi*(0:127)/16);\n[pxx, w] = periodogram(x, [], 128, 'twosided');", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(pxx,'double')); assert(size(pxx,1)==numel(w));" } },
        BuiltinExample { id: "example-4", title: "Column-wise spectra for a matrix", program: "x = [sin(2*pi*(0:255)'/16), cos(2*pi*(0:255)'/32)];\n[pxx, f] = periodogram(x, [], 256, 1000);", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(pxx,'double')); assert(size(pxx,1)==numel(f));" } },
        BuiltinExample { id: "example-5", title: "Evaluate at explicit normalized frequencies", program: "w = linspace(0, pi, 64);\n[pxx, wout] = periodogram(randn(1,512), [], w);", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(pxx,'double')); assert(size(pxx,1)==numel(wout));" } }],
    example_exemption: None,

    faqs: &[BuiltinDocumentationFaq { question: "How is `periodogram` different from `pwelch`?", answer: "`periodogram` estimates a spectrum from one windowed full-signal segment. `pwelch` splits the signal into overlapped segments and averages the modified periodograms." }, BuiltinDocumentationFaq { question: "Does `periodogram` include the Nyquist bin?", answer: "For one-sided output and even `nfft`, yes. The output length is `nfft/2 + 1`, with DC first and the Nyquist bin last." }, BuiltinDocumentationFaq { question: "Does `periodogram` plot when no outputs are requested?", answer: "RunMat currently returns values using normal output-count semantics instead of opening a plot. Use the returned `pxx` and frequency vector with plotting functions explicitly." }],

    links: &[BuiltinDocumentationLink { label: "pwelch", target: BuiltinDocumentationLinkTarget::Builtin("pwelch") }, BuiltinDocumentationLink { label: "spectrogram", target: BuiltinDocumentationLinkTarget::Builtin("spectrogram") }, BuiltinDocumentationLink { label: "fft", target: BuiltinDocumentationLinkTarget::Builtin("fft") }, BuiltinDocumentationLink { label: "hamming", target: BuiltinDocumentationLinkTarget::Builtin("hamming") }, BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/periodogram.rs") }],
    media: &[],

    evidence: BuiltinDocumentationEvidence {
         implementation: &[BuiltinDocumentationLink { label: "Runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/periodogram.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Signal-class matrix", location: "crates/runmat-runtime/tests/spectral_signal_precision.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Owned spectral publication and restoration", location: "crates/runmat-runtime/tests/signal_output_ownership.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Existing numerical and argument semantics", location: "builtins::math::signal::periodogram::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Existing integration coverage", location: "crates/runmat-vm/tests/integer_numeric_transform_semantics.rs" }],
        notes: &["Signal precision is a RunMat output policy; no verified MATLAB native-class parity is claimed. Host FFT/window arithmetic remains binary64 with one final Single publication cast. Surplus requested slots remain shared host Double zero padding; spectrogram reassignment is not supported."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
