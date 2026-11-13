use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("sawtooth"),
    slug: Some("sawtooth"),
    summary: "Generate a periodic sawtooth waveform with optional peak position.",
    description: "`y = sawtooth(t)` generates a periodic sawtooth wave with period `2*pi` at sample times `t`, ranging in `[-1, 1]`. `y = sawtooth(t, xmax)` controls the peak position using `xmax` in `[0, 1]`, following MATLAB-compatible waveform conventions.",
    keywords: &["sawtooth", "waveform", "signal processing", "triangle", "periodic", "elementwise"],
    related: &["square", "sin", "cos", "linspace"],
    sections: &[
        BuiltinDocumentationSection { heading: "Waveform and input", paragraphs: &[
            "Evaluation is element-wise for real scalar and N-D sample times and retains their shape. The phase is reduced modulo `2*pi` for positive and negative times. At the default `xmax = 1`, the waveform rises from `-1` and wraps to `-1` at the next period boundary.",
            "At `xmax = 0.5` the piecewise-linear wave peaks at one halfway through each period. `xmax = 0` selects a falling ramp; `xmax = 1` selects a rising ramp. The control must be finite, scalar, and in `[0, 1]`.",
            "Non-finite sample times yield NaN; complex and text inputs are rejected. Double sample times are the documented compatibility form. Single, integer, and logical inputs are RunMat extensions and produce double output; typed integer controls require exact conversion at the binary64 waveform boundary.",
        ]},
        BuiltinDocumentationSection { heading: "Resident data", paragraphs: &[
            "The current runtime has no sawtooth GPU kernel or fusion path. Interactive gpuArray input is a RunMat-only extension; admitted resident data is downloaded through its owning provider and the result is returned on the host. Automatic placement must account for this transfer.",
        ]},
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Default rising sawtooth over two periods", program: "t = 0:pi/2:4*pi;\ny = sawtooth(t)", display_output: Some("y = [-1 -0.5 0 0.5 -1 -0.5 0 0.5 -1]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(y),size(t))); assert(max(abs(y-[-1 -0.5 0 0.5 -1 -0.5 0 0.5 -1]))<1e-12);" } },
        BuiltinExample { id: "example-2", title: "Triangle wave via xmax = 0.5", program: "t = linspace(0, 2*pi, 5);\ny = sawtooth(t, 0.5)", display_output: Some("y = [-1 0 1 0 -1]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(max(abs(y-[-1 0 1 0 -1]))<1e-12);" } },
        BuiltinExample { id: "example-3", title: "Pure falling sawtooth via xmax = 0", program: "y = sawtooth([0 pi 2*pi], 0)", display_output: Some("y = [1 0 1]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(max(abs(y-[1 0 1]))<1e-12);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Why does `sawtooth(2*pi)` return `-1` instead of `+1`?", answer: "RunMat reduces the input modulo `2*pi`, so `2*pi` is folded back to the start of the next period. The default rising sawtooth therefore restarts at `-1` exactly on each period boundary, matching MATLAB." },
        BuiltinDocumentationFaq { question: "How do I produce a triangle wave?", answer: "Pass `xmax = 0.5` so the peak sits at the centre of each period and the waveform ramps symmetrically up and back down." },
        BuiltinDocumentationFaq { question: "Can `sawtooth` accept complex inputs?", answer: "No. `sawtooth` is defined on real samples and rejects complex inputs with a builtin-scoped error to match MATLAB." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "square", target: BuiltinDocumentationLinkTarget::Builtin("square") },
        BuiltinDocumentationLink { label: "sin", target: BuiltinDocumentationLinkTarget::Builtin("sin") },
        BuiltinDocumentationLink { label: "cos", target: BuiltinDocumentationLinkTarget::Builtin("cos") },
        BuiltinDocumentationLink { label: "linspace", target: BuiltinDocumentationLinkTarget::Builtin("linspace") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/sawtooth.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Sawtooth phase, class, and host fallback", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/sawtooth.rs") }, BuiltinDocumentationLink { label: "Shared typed waveform scalar controls", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/common/waveform_control.rs") }, BuiltinDocumentationLink { label: "Typed, cooperative host waveform mapping", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/common/waveform_map.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Period, peak, classes, and provider behavior", location: "builtins::math::signal::sawtooth::tests" }],
        notes: &[],
    },
    introduced: None,
    status: None,
};
