use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/step.rs"),
};
const SOLVER_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Step response solver",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/response/step/solver.rs"),
};
const TIME_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Step time grids",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/response/step/time.rs"),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("step"), slug: Some("step"),
    summary: "Compute or plot the unit-step response of a SISO transfer function.",
    description: "`step(sys)` samples the response to a unit step. A positive finite scalar sets the final time; a real vector supplies exact sample times. With outputs, the operation returns an N-by-1 double response and, optionally, its time coordinates. Without outputs it plots the response.",
    keywords: &["step", "step response", "control system", "transfer function", "tf"],
    related: &["tf", "plot", "stepinfo"],
    sections: &[
        BuiltinDocumentationSection { heading: "Models and time", paragraphs: &[
            "Supported models are real SISO tf objects with proper numerator/denominator degree, a known nonnegative sample time, and zero input/output delay. State-space, MIMO, complex-coefficient, delayed and improper models are not yet simulated. The response of a discrete-time model uses its stored Ts.",
            "With no time argument RunMat selects a grid. A scalar final time is positive and finite; an explicit vector is finite, nonnegative, and nondecreasing. Discrete-time samples must align with Ts. Output response and time coordinates are double column vectors.",
            "No-output calls also accept multiple model/style pairs followed by one time argument, for example `step(sys1, 'b', sys2, 'r--', 5)`. Style arguments belong to the preceding model. A malformed model or time argument is rejected before plotting begins.",
            "Typed-integer time input is a RunMat-only extension. Typed-integer tf coefficients or Ts and logical tf coefficients assigned after construction use separate model-property gates. Integer time and model-property values must be exactly representable in binary64 at the host simulation boundary; wide integers that would round are rejected. The MATLAB compatibility mode rejects these extensions before provider access.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
            "Response simulation is host-side and ends fusion. Required resident tf properties and time input gather through their owning provider after locally decidable validation; unrelated object properties are not consumed. Returned numeric arrays are host-resident.",
            "No-output calls may use the user-interface capability to produce a plot. Output-producing calls have no plotting effect. Native and browser execution depend on the available plotting service; this contract does not claim a GPU response kernel.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "first-order", title: "First-order response", program: "sys = tf(1, [1 1]); [y,t] = step(sys, [0 1 2]); fprintf('%d %d %d %d\\n', size(y,1), size(y,2), size(t,1), size(t,2));", display_output: Some("3 1 3 1"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(y), [3 1])); assert(isequal(t, [0;1;2])); assert(abs(y(2)-(1-exp(-1))) < 1e-4);" } },
        BuiltinExample { id: "final-time", title: "Scalar final time", program: "sys = tf(1, [1 1]); [y,t] = step(sys, 5); fprintf('t(end) = %g\\n', t(end));", display_output: Some("t(end) = 5"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(y), size(t))); assert(abs(t(end)-5) < 1e-12);" } },
        BuiltinExample { id: "discrete", title: "Discrete response", program: "sys = tf(1, [1 -0.5], 0.1); [y,t] = step(sys, [0 0.1 0.2]);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(t, [0;0.1;0.2])); assert(max(abs(y-[0;1;1.5])) < 1e-10);" } },
        BuiltinExample { id: "integer-time", title: "RunMat integer time", program: "sys = tf(1, [1 1]); [y,t] = step(sys, uint16([0 1 2]));", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(t, [0;1;2])); assert(strcmp(class(y), 'double'));" } },
        BuiltinExample { id: "plot", title: "Plot a step response", program: "sys = tf(1, [1 1]); step(sys); fprintf('ok\\n');", display_output: Some("ok"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isequal(size(step(sys, [0 1])), [2 1]));" } },
        BuiltinExample { id: "multi-plot", title: "Plot two systems", program: "H1 = tf(1, [1 1]); H2 = tf(2, [1 2]); step(H1, 'b', H2, 'r--', 5); fprintf('ok\\n');", display_output: Some("ok"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isequal(size(step(H1, [0 1])), [2 1]));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does step support MIMO models?", answer: "Not yet. The current response simulator supports SISO transfer-function objects." },
        BuiltinDocumentationFaq { question: "Does step support state-space models?", answer: "Not yet. The current response simulator accepts tf objects; ss support remains outside this implementation." },
        BuiltinDocumentationFaq { question: "Can coefficients be complex?", answer: "No. Non-real tf coefficients return an unsupported-model diagnostic." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "tf", target: BuiltinDocumentationLinkTarget::Builtin("tf") },
        BuiltinDocumentationLink { label: "plot", target: BuiltinDocumentationLinkTarget::Builtin("plot") },
        BuiltinDocumentationLink { label: "stepinfo", target: BuiltinDocumentationLinkTarget::Builtin("stepinfo") },
        SOURCE,
        SOLVER_SOURCE,
        TIME_SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE, SOLVER_SOURCE, TIME_SOURCE], verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Step response, time, compatibility and admission", location: "crates/runmat-runtime/src/builtins/control/step.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM tf/step workflow", location: "crates/runmat-vm/tests/control.rs" },
    ], notes: &["The simulator supports a bounded real SISO tf subset; browser and provider execution require integrated qualification."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
