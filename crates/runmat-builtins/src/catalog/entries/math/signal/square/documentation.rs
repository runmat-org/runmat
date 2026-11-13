use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("square"),
    slug: Some("square"),
    summary: "Generate periodic square waves with MATLAB-compatible duty-cycle control.",
    description: "`y = square(t)` generates a period-`2*pi` square wave sampled at `t` with outputs in `{-1, +1}`. `y = square(t, duty)` applies MATLAB-compatible percentage duty-cycle control over `[0, 100]`.",
    keywords: &["square", "waveform", "signal processing", "duty cycle", "periodic", "elementwise"],
    related: &["sawtooth", "sin", "cos", "linspace"],
    sections: &[
        BuiltinDocumentationSection { heading: "Waveform and input", paragraphs: &[
            "Evaluation is element-wise for real scalar and N-D sample times, retaining their shape. The phase is reduced modulo `2*pi` for positive and negative times. Output is `+1` when `phi/(2*pi) < duty/100` and `-1` at and after the half-open boundary.",
            "Duty zero produces constant `-1`; duty 100 produces constant `+1`. Duty must be a finite scalar in `[0, 100]`. Non-finite sample times yield NaN; complex and text input is rejected with a builtin-scoped error.",
            "Single and double sample times retain their floating output class. Integer and logical sample times or controls are RunMat extensions. Integer sample times must cross the checked binary64 phase boundary; integer input yields double output.",
        ]},
        BuiltinDocumentationSection { heading: "Resident data", paragraphs: &[
            "The current runtime has no square-wave GPU kernel or fusion path. Explicit gpuArray input is a RunMat-only extension; admitted resident data is downloaded through its owning provider and the result is returned on the host. Automatic placement must account for this transfer.",
        ]},
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Default 50% duty cycle square wave", program: "t = 0:pi/4:2*pi;\ny = square(t)", display_output: Some("y = [1 1 1 1 -1 -1 -1 -1 1]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(y),size(t))); assert(isequal(y,[1 1 1 1 -1 -1 -1 -1 1]));" } },
        BuiltinExample { id: "example-2", title: "25% duty cycle square wave", program: "t = 0:pi/4:2*pi;\ny = square(t, 25)", display_output: Some("y = [1 1 -1 -1 -1 -1 -1 -1 1]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(y),size(t))); assert(isequal(y,[1 1 -1 -1 -1 -1 -1 -1 1]));" } },
        BuiltinExample { id: "example-3", title: "Constant outputs at the boundaries", program: "y = square([0 pi 2*pi], 100)", display_output: Some("y = [1 1 1]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(y,[1 1 1])); assert(isequal(square([0 pi 2*pi],0),[-1 -1 -1]));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What happens exactly at the duty-cycle boundary?", answer: "RunMat uses half-open semantics: the boundary phase belongs to the negative half. With the default 50% duty cycle, `square(pi)` returns `-1`. This matches MATLAB's documented behaviour." },
        BuiltinDocumentationFaq { question: "How do I produce a constant `+1` or `-1`?", answer: "Pass `duty = 100` for a constant `+1` waveform and `duty = 0` for a constant `-1` waveform." },
        BuiltinDocumentationFaq { question: "Can `square` accept complex inputs?", answer: "No. `square` is defined on real samples and rejects complex inputs with a builtin-scoped error to match MATLAB." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "sawtooth", target: BuiltinDocumentationLinkTarget::Builtin("sawtooth") },
        BuiltinDocumentationLink { label: "sin", target: BuiltinDocumentationLinkTarget::Builtin("sin") },
        BuiltinDocumentationLink { label: "cos", target: BuiltinDocumentationLinkTarget::Builtin("cos") },
        BuiltinDocumentationLink { label: "linspace", target: BuiltinDocumentationLinkTarget::Builtin("linspace") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/square.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Square-wave boundary, class, and host fallback", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/square.rs") }, BuiltinDocumentationLink { label: "Shared typed waveform scalar controls", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/common/waveform_control.rs") }, BuiltinDocumentationLink { label: "Typed, cooperative host waveform mapping", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/signal/common/waveform_map.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Duty boundary, classes, and provider behavior", location: "builtins::math::signal::square::tests" }],
        notes: &[],
    },
    introduced: None,
    status: None,
};
