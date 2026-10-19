use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/nyquist.rs"),
};
const ADMISSION_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Nyquist admission",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/nyquist/admission.rs"),
};
const FREQUENCY_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Nyquist frequency grid",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/nyquist/frequency.rs"),
};
const SOLVER_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Nyquist solver",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/nyquist/solver.rs"),
};
const PLOT_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Nyquist plot",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/nyquist/plot.rs"),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("nyquist"), slug: Some("nyquist"),
    summary: "Compute or plot the Nyquist frequency response of a SISO transfer function.",
    description: "`nyquist(sys)` plots a supported scalar SISO `tf` model when outputs are discarded. Output forms return real, imaginary and frequency columns for the default or explicit frequency grid. Continuous models are evaluated at `s = jω`; discrete models use `z = exp(jωTs)`. State-space, FRD, model arrays and MIMO inputs are not yet supported.",
    keywords: &["nyquist", "nyquist plot", "frequency response", "control system", "transfer function", "tf"],
    related: &["tf", "plot", "pole"],
    sections: &[
        BuiltinDocumentationSection { heading: "Frequency grid and outputs", paragraphs: &[
            "An explicit w must be a nonempty vector of finite nonnegative angular frequencies. With no w, a continuous model uses a deterministic logarithmic grid based on poles and zeros; a discrete model uses 200 points inside zero and the Nyquist frequency π/Ts, excluding potentially singular endpoints. These grid choices describe RunMat behavior, not verified MATLAB point-for-point parity.",
            "`re`, `im` and `wout` are same-length double N-by-1 columns. One output returns re, two return re and im, and three add wout. A no-output call plots the positive-frequency curve and, for exactly real coefficients, mirrors the negative-frequency curve. Complex-coefficient models are evaluated without a mirrored curve. An exact denominator zero at a requested frequency produces an explicit singularity error.",
        ] },
        BuiltinDocumentationSection { heading: "Compatibility and placement", paragraphs: &[
            "Native typed-integer frequency/model metadata, logical numeric roles, and GPU-resident consumed values are conservative RunMat-only extensions. Typed integers must be exactly representable in binary64 before complex frequency evaluation. Delayed transfer functions are not supported; nonzero input or output delay is rejected exactly rather than suppressed by a tolerance.",
            "Frequency evaluation and output materialization run on the host. Valid consumed resident coefficient or frequency values gather through their owning provider only after host-known form, compatibility and structural checks; unrelated model properties are not read. The numeric output path has no UI effect; plotting needs the UI service. Calls may suspend and throw, but have no workspace, filesystem, network, callback, randomness or time effect.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "explicit", title: "Compute response data at explicit frequencies", program: "H = tf(1, [1 1]); w = [0 1 2]; [re, im, wout] = nyquist(H, w); fprintf(\"%.1f %.1f %.1f %.1f\\n\", re(1), re(2), im(2), wout(3));", display_output: Some("1.0 0.5 -0.5 2.0"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(re(1)-1) < 1e-10); assert(abs(re(2)-.5) < 1e-10); assert(abs(im(2)+.5) < 1e-10); assert(wout(3)==2);" } },
        BuiltinExample { id: "discrete", title: "Response of a sampled transfer function", program: "H = tf(1, [1 -.5], .1); [re, im, w] = nyquist(H, [0 1]); fprintf(\"%.1f %.1f\\n\", re(1), im(1));", display_output: Some("2.0 0.0"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(re(1)-2) < 1e-10); assert(abs(im(1)) < 1e-10); assert(numel(w)==2);" } },
        BuiltinExample { id: "plot", title: "Plot a Nyquist diagram", program: "H = tf(1, [1 2 1]); nyquist(H); fprintf(\"ok\\n\");", display_output: Some("ok"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isgraphics(gcf()));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does nyquist accept MIMO or state-space models?", answer: "Not yet. The current implementation accepts scalar SISO tf objects." },
        BuiltinDocumentationFaq { question: "Does nyquist run on a GPU?", answer: "No. The calculation is host-side; admitted consumed resident input values gather through their owner." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "tf", target: BuiltinDocumentationLinkTarget::Builtin("tf") },
        BuiltinDocumentationLink { label: "plot", target: BuiltinDocumentationLinkTarget::Builtin("plot") },
        SOURCE,
        ADMISSION_SOURCE,
        FREQUENCY_SOURCE,
        SOLVER_SOURCE,
        PLOT_SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE, ADMISSION_SOURCE, FREQUENCY_SOURCE, SOLVER_SOURCE, PLOT_SOURCE], verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Continuous/discrete/complex response, errors, output counts and admission", location: "crates/runmat-runtime/src/builtins/control/nyquist.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM control workflows", location: "crates/runmat-vm/tests/control.rs" },
    ], notes: &["Point-for-point MATLAB default-grid and all-scale polynomial numerical parity are not asserted; native/browser/provider qualification belongs to the integrated batch."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
