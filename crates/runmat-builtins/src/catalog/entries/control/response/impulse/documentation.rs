use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/impulse.rs"),
};
const SOLVER_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Impulse response solver",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/impulse/solver.rs"),
};
const TIME_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Impulse time grids",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/impulse/time.rs"),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("impulse"), slug: Some("impulse"),
    summary: "Compute or plot the impulse response of a SISO transfer function.",
    description: "`impulse(sys)` computes the impulse response of a supported dynamic system. With output arguments it returns numeric samples; without outputs it plots the response when a plotting service is available.",
    keywords: &["impulse", "impulse response", "control system", "transfer function", "tf"],
    related: &["tf", "plot", "stem"],
    sections: &[
        BuiltinDocumentationSection { heading: "Models and time", paragraphs: &[
            "The current simulator accepts real, strictly proper SISO tf objects created by `tf(num, den)` or `tf(num, den, Ts)`. It does not yet support MIMO, ss, zpk, frd, model arrays, identified-system families, complex coefficients, nonzero delays, static gains, or non-strictly-proper transfer functions. The last two forms include a direct impulse term that is not represented as a finite response vector.",
            "With no time argument, RunMat chooses a deterministic time vector. A finite nonnegative scalar generates a grid from zero to the final time; a real finite nonnegative strictly increasing vector supplies exact time coordinates. Continuous-time responses use a controllable canonical realization. Discrete-time responses use a unit-area impulse of height 1/Ts; explicit times must be integer multiples of Ts.",
            "The response and time outputs are N-by-1 double columns. Typed-integer, logical, single, and resident numeric roles are separately gated RunMat extensions, including numeric metadata consumed from a tf object. Typed integers must cross the floating simulation boundary exactly.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
            "Simulation runs on the host. Required resident model properties and time input gather through their owning provider after host-known input validation; unrelated object properties are not consumed. Output arrays are host resident and the operation ends fusion.",
            "No-output calls may use the user-interface capability and plot continuous systems with plot or discrete systems with stem. Output-producing calls have no plotting effect. GPU kernels do not evaluate the response.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "first-order", title: "First-order impulse response", program: "H = tf(20, [1 5]); [y,t] = impulse(H, [0 0.1 0.2]); fprintf('%d %d %d %d\\n', size(y,1), size(y,2), size(t,1), size(t,2));", display_output: Some("3 1 3 1"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(t, [0;0.1;0.2])); assert(isequal(size(y), [3 1])); assert(max(abs(y-[20;20*exp(-0.5);20*exp(-1)])) < 1e-8);" } },
        BuiltinExample { id: "final-time", title: "Scalar final time", program: "H = tf(1, [1 3 2]); [y,t] = impulse(H, 2);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(y), size(t))); assert(abs(t(end)-2) < 1e-12);" } },
        BuiltinExample { id: "discrete", title: "Discrete-time impulse response", program: "H = tf(1, [1 -0.5], 0.1); [y,t] = impulse(H, [0 0.1 0.2]);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(t, [0;0.1;0.2])); assert(max(abs(y-[0;10;5])) < 1e-10);" } },
        BuiltinExample { id: "plot", title: "Plot an impulse response", program: "H = tf(1, [1 3 2]); impulse(H); fprintf('ok\\n');", display_output: Some("ok"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "[y,t] = impulse(H, [0 1]); assert(isequal(size(y), [2 1]));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does impulse support MIMO systems?", answer: "Not yet. The current simulator supports real SISO tf objects." },
        BuiltinDocumentationFaq { question: "Why are non-strictly-proper transfer functions rejected?", answer: "They include a direct impulse term that cannot be represented by this implementation's finite numeric response vector." },
        BuiltinDocumentationFaq { question: "Can I use state-space or zero-pole-gain models?", answer: "Not yet. ss, zpk, frd, model arrays, and identified-system families are outside the current implementation." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "tf", target: BuiltinDocumentationLinkTarget::Builtin("tf") },
        BuiltinDocumentationLink { label: "plot", target: BuiltinDocumentationLinkTarget::Builtin("plot") },
        BuiltinDocumentationLink { label: "stem", target: BuiltinDocumentationLinkTarget::Builtin("stem") },
        SOURCE,
        SOLVER_SOURCE,
        TIME_SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE, SOLVER_SOURCE, TIME_SOURCE], verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Impulse response, time, compatibility and admission", location: "crates/runmat-runtime/src/builtins/control/impulse.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM tf/impulse workflow", location: "crates/runmat-vm/tests/control.rs" },
    ], notes: &["The simulator supports a bounded real strictly proper SISO tf subset; browser and provider execution require integrated qualification."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
