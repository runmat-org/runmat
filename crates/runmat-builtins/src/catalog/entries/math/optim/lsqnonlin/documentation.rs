use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {

    authority: BuiltinDocumentationAuthority::Catalog,

    title: Some("lsqnonlin"),
 slug: Some("lsqnonlin"),

    summary: "Solve bounded and unbounded nonlinear least-squares residual problems.",

    description: "`lsqnonlin` minimizes the squared 2-norm of a vector-valued residual callback `fun(x)` using RunMat's shared finite-difference Levenberg-Marquardt least-squares solver. Scalar, vector, and matrix `x0` inputs are supported and the returned solution preserves the shape of `x0`.",

    keywords: &["lsqnonlin",
"least squares",
"optimization",
"bounds",
"jacobian"],

    related: &["lsqcurvefit",
"fsolve",
"optimoptions",
"optimset"],

    sections: &[BuiltinDocumentationSection {
 heading: "Behavior and supported forms",
 paragraphs: &["The residual callback is evaluated as `fun(x)` and must return a finite real numeric scalar, vector, or array. The residual output preserves the callback's initial shape.",
"Lower and upper bounds may be empty or contain up to `numel(x0)` elements. Supplied bounds apply to the leading variables; omitted trailing entries are unbounded, matching MATLAB's partial-bound convention.",
"`optimset` and `optimoptions` structs are accepted, and `[]` option placeholders are treated as defaults. `TolX`, `TolFun`, `MaxIter`, `MaxFunEvals`, `Display`, and `Algorithm` are recognized.",
"RunMat accepts `Algorithm` values `trust-region-reflective` and `levenberg-marquardt` for compatibility metadata, but both execute through the shared finite-difference Levenberg-Marquardt host solver.",
"Linear and nonlinear constraints require MATLAB's newer interior-point constrained least-squares path. RunMat accepts empty placeholders in those argument positions but rejects non-empty constraints with a deterministic error.",
"Problem structs are supported when they provide `solver`, `objective` or `fun`, `x0`, optional `lb`, optional `ub`, and optional `options`. Constraint fields including `A`, `b`, `Aineq`, `bineq`, `Aeq`, `beq`, and `nonlcon` must be absent or empty.",
"The `output` struct includes `iterations`, `funcCount`, `firstorderopt`, `stepsize`, `cgiterations`, `lssteplength`, `algorithm`, and `message`.",
"Syntax: x = lsqnonlin(fun, x0)",
"Syntax: x = lsqnonlin(fun, x0, lb, ub)",
"Syntax: x = lsqnonlin(fun, x0, lb, ub, options)",
"Syntax: x = lsqnonlin(problem)",
"Syntax: [x, resnorm] = lsqnonlin(___)",
"Syntax: [x, resnorm, residual] = lsqnonlin(___)",
"Syntax: [x, resnorm, residual, exitflag] = lsqnonlin(___)",
"Syntax: [x, resnorm, residual, exitflag, output] = lsqnonlin(___)",
"Syntax: [x, resnorm, residual, exitflag, output, lambda] = lsqnonlin(___)",
"Syntax: [x, resnorm, residual, exitflag, output, lambda, jacobian] = lsqnonlin(___)",
"Syntax: x = lsqnonlin(fun, x0, lb, ub, [], [], [], [], [], options)",
"Existing RunMat integer and logical solver data is normalized into host binary64, including rounding outside its exact integer range. This is not an exact-integer solver or a claim of MATLAB integer-overload parity. Structural count options retain exact checked decoding."] }
,

BuiltinDocumentationSection {
 heading: "Execution and residency",
 paragraphs: &["`lsqnonlin` gathers residuals and bounds to the host for iterative solving. Callback bodies can still use GPU-aware builtins; provider-native nonlinear least-squares is deferred to the GPU fast-path audit.",
"The solver state uses host binary64; there is no provider-native solver kernel or numeric fusion producer. Callback computations retain their own capabilities, effects and acceleration policies. Native and browser routes use the same algorithm and the owning execution context's cooperative CPU budget.",
"Cancellation is checked between bounded input/result scans and solver steps. Native dense linear algebra calls, where used, run to completion between checks rather than being hard-preempted."] }
,

BuiltinDocumentationSection {
 heading: "Example notes",
 paragraphs: &["Fit a vector residual: x = [2; 3]",
"Request diagnostics: x approaches [1 2], resnorm approaches 0, and jacobian is 2-by-2."] }
],

    examples: &[BuiltinExample {
 id: "example-1",
 title: "Fit a vector residual",
 program: "r = @(x) [x(1)-2; x(2)-3; x(1)+x(2)-5];\nx = lsqnonlin(r, [0; 0])",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(norm(x-[2;3])<1e-4);" }
 }
,

BuiltinExample {
 id: "example-2",
 title: "Use bounds",
 program: "x = lsqnonlin(@(x) [x-2, x-2], 0, -1, 1)",
 display_output: Some("x = 1"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(abs(x-1)<1e-5);" }
 }
,

BuiltinExample {
 id: "example-3",
 title: "Request diagnostics",
 program: "opts = optimoptions(\"lsqnonlin\", \"Algorithm\", \"levenberg-marquardt\", \"Display\", \"off\");\n[x, resnorm, residual, exitflag, output, lambda, jacobian] = lsqnonlin(@(x) [x(1)-1, 2*x(2)-4], [0 0], [], [], opts)",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(norm(x-[1 2])<1e-4); assert(resnorm<1e-8); assert(isequal(size(jacobian),[2 2])); assert(isstruct(lambda)); assert(isstruct(output));" }
 }
],

    example_exemption: None,

    faqs: &[BuiltinDocumentationFaq {
 question: "Does lsqnonlin support linear or nonlinear constraints?",
 answer: "Not yet. Empty A, b, Aineq, bineq, Aeq, beq, and nonlcon placeholders are accepted so common option-forwarding scripts run, but non-empty constraints are rejected because RunMat does not yet have the interior-point constrained least-squares infrastructure." }
,

BuiltinDocumentationFaq {
 question: "Does lsqnonlin run on the GPU?",
 answer: "The solver is host iterative work. Callback code may call GPU-aware builtins, but residuals and bounds are gathered for the solver. Provider-native nonlinear least-squares execution belongs to the queued GPU fast-path audit rather than this compatibility slice." }
],

    links: &[BuiltinDocumentationLink {
 label: "lsqcurvefit",
 target: BuiltinDocumentationLinkTarget::Builtin("lsqcurvefit") }
,

BuiltinDocumentationLink {
 label: "fsolve",
 target: BuiltinDocumentationLinkTarget::Builtin("fsolve") }
,

BuiltinDocumentationLink {
 label: "optimoptions",
 target: BuiltinDocumentationLinkTarget::Builtin("optimoptions") }
,

BuiltinDocumentationLink {
 label: "optimset",
 target: BuiltinDocumentationLinkTarget::Builtin("optimset") }
,

BuiltinDocumentationLink {
 label: "Implementation",
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/lsqnonlin.rs") }
],

    media: &[],

    evidence: BuiltinDocumentationEvidence {

        implementation: &[
            BuiltinDocumentationLink {
 label: "Solver callback and numeric preparation",
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/common.rs") }
,

            BuiltinDocumentationLink {
 label: "Shared finite-difference solver",
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/least_squares.rs") }
,

        ],

        verification: &[BuiltinEvidenceReference {
 kind: BuiltinEvidenceKind::UnitTest,
 label: "Retained unit source",
 location: "builtins::math::optim::lsqnonlin::tests" }
,
BuiltinEvidenceReference {
 kind: BuiltinEvidenceKind::IntegrationTest,
 label: "Retained integration source",
 location: "runmat-vm/tests/closures.rs::lsqnonlin_accepts_anonymous_residual_function" }
],

        notes: &["Source references are not execution receipts. Native, browser, callback, residency and standalone examples require integrated qualification."],

    }
,

    introduced: Some("eval-loop"),

    status: Some(BuiltinDocumentationStatus::Partial),

}
;
