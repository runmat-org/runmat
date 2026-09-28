use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,

    title: Some("spectrogram"),
    slug: Some("spectrogram"),
    summary: "Compute a short-time Fourier transform spectrogram.",
    description: "`spectrogram(x)` computes a short-time Fourier transform (STFT) of a signal. Optional arguments select the segment window, overlap, DFT length or explicit frequency grid, sampling frequency, frequency range, and PSD or power-spectrum scaling.",

    keywords: &["spectrogram", "STFT", "short-time Fourier transform", "PSD", "power spectrum", "time frequency", "signal processing"],
    related: &["pwelch", "periodogram", "fft", "hamming"],

    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &["RunMat returns the computed S, F, T, and PS outputs in the signal's floating precision: single signals produce single STFT, PSD, frequency, and time arrays, and double signals produce double arrays. STFT storage remains complex. Window and sampling-control classes do not change this rule. This is RunMat's policy, not a claim of verified MATLAB class parity. The host reference computes FFT and window arithmetic in double and casts once when publishing single outputs.",
                "`s = spectrogram(x)` returns the complex STFT matrix with time increasing across columns and frequency increasing down rows.",
                "`[s, f, t, ps] = spectrogram(...)` also returns the frequency vector, segment midpoint vector, and PSD or power-spectrum matrix.",
                "`win` may be a positive scalar length, a finite real window vector, or `[]`. `[]` uses a Hamming window sized to divide the signal into about eight 50%-overlapped sections.",
                "`nOverlap` defaults to half the window length and must be less than the window length.",
                "`freqSpec` defaults to `max(256, 2^nextpow2(length(win)))`. A real frequency vector with at least two elements may be supplied to evaluate explicit frequencies.",
                "Omitting `Fs` uses normalized radians/sample and segment midpoint times divided by `2*pi`. Passing `Fs` as `[]` uses a 1 Hz sample rate.",
                "Real inputs default to one-sided output; complex inputs default to two-sided output and reject `'onesided'`.",
                "The optional range selector accepts `'onesided'`, `'twosided'`, or `'centered'`. Range selectors are not combined with explicit frequency vectors.",
                "The optional spectrum type accepts `'psd'` (default) or `'power'`; `'power'` uses coherent-gain normalization.",
                "`'yaxis'` is accepted as a plotting-orientation no-op for numeric execution. `'reassigned'` is rejected until true energy-center reassignment is implemented.",
                "The documented signal storage class is single or double. Typed integer signals, windows, overlap counts, frequency specifications, and sample rates are role-specific RunMat extensions. Any value entering binary64 rejects if it would round."],
        },
        BuiltinDocumentationSection { heading: "Execution and ownership", paragraphs: &["Resident routes use the signal's exact owner and native class. Typed unsupported capabilities select the reference path; operational failures retain their source and do not replay execution.", "Requested computed outputs use checked class-preserving restoration after reference evaluation. Explicit placement requires successful restoration; automatic placement may return a host output when restoration is unsupported. Original inputs and earlier outputs remain protected.", "Uniform-grid spectrograms keep GPU-resident signal data on device through windowing, framing, FFT, frequency selection, and power scaling. Explicit-frequency and unsupported option forms fall back to host execution."] },
    ],

    examples: &[
        BuiltinExample { id: "example-1", title: "STFT of a sine wave", program: "fs = 1000;\nt = (0:999) / fs;\nx = sin(2*pi*100*t);\n[s, f, tt, ps] = spectrogram(x, hamming(128), 64, 256, fs);", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(s,'double')); assert(size(s,1)==numel(f)); assert(size(s,2)==numel(tt));" } },
        BuiltinExample { id: "example-2", title: "Use scalar window length", program: "[s, f, t] = spectrogram(randn(1,1024), 128, 64, 256);", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(s,'double')); assert(size(s,1)==numel(f)); assert(size(s,2)==numel(t));" } },
        BuiltinExample { id: "example-3", title: "Centered complex STFT", program: "x = exp(1i*2*pi*(0:127)/16);\n[s, f, t] = spectrogram(x, 32, 16, 64, 'centered');", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(s,'double')); assert(size(s,1)==numel(f)); assert(size(s,2)==numel(t));" } },
        BuiltinExample { id: "example-4", title: "Power spectrum scaling", program: "[s, f, t, p] = spectrogram(randn(1,512), 64, 32, 128, 'power');", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(s,'double')); assert(size(s,1)==numel(f)); assert(size(s,2)==numel(t));" } },
        BuiltinExample { id: "example-5", title: "Explicit normalized frequencies", program: "w = linspace(0, pi, 64);\n[s, wout, t] = spectrogram(randn(1,512), hamming(128), 64, w);", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(s,'double')); assert(size(s,1)==numel(wout)); assert(size(s,2)==numel(t));" } }],
    example_exemption: None,

    faqs: &[BuiltinDocumentationFaq { question: "What is the relationship between `s` and `ps`?", answer: "`s` is the complex STFT. `ps` is derived from `abs(s).^2` using PSD or power-spectrum normalization and one-sided doubling when appropriate." }, BuiltinDocumentationFaq { question: "Does `spectrogram` plot when no outputs are requested?", answer: "RunMat currently returns values using normal output-count semantics instead of opening a plot. The `'yaxis'` token is accepted so scripts can continue numeric execution." }, BuiltinDocumentationFaq { question: "Does RunMat implement reassigned spectrogram energy centers?", answer: "No. RunMat rejects `'reassigned'` rather than returning unreassigned coordinate grids." }],

    links: &[BuiltinDocumentationLink { label: "pwelch", target: BuiltinDocumentationLinkTarget::Builtin("pwelch") }, BuiltinDocumentationLink { label: "periodogram", target: BuiltinDocumentationLinkTarget::Builtin("periodogram") }, BuiltinDocumentationLink { label: "fft", target: BuiltinDocumentationLinkTarget::Builtin("fft") }, BuiltinDocumentationLink { label: "hamming", target: BuiltinDocumentationLinkTarget::Builtin("hamming") }, BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/spectrogram.rs") }],
    media: &[],

    evidence: BuiltinDocumentationEvidence {
         implementation: &[BuiltinDocumentationLink { label: "Runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/spectrogram.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Signal-class matrix", location: "crates/runmat-runtime/tests/spectral_signal_precision.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Owned spectral publication and restoration", location: "crates/runmat-runtime/tests/signal_output_ownership.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Existing numerical and argument semantics", location: "builtins::math::signal::spectrogram::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Existing integration coverage", location: "crates/runmat-vm/tests/basics.rs" }],
        notes: &["Signal precision is a RunMat output policy; no verified MATLAB native-class parity is claimed. Host FFT/window arithmetic remains binary64 with one final Single publication cast. Surplus requested slots remain shared host Double zero padding; spectrogram reassignment is not supported."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
