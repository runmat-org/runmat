use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {

    authority: BuiltinDocumentationAuthority::Catalog,

    title: Some("fsolve"),
 slug: Some("fsolve"),

    summary: "Solve nonlinear equation systems in MATLAB and RunMat.",

    description: "`x = fsolve(fun, x0)` solves `fun(x) = 0` starting from binary64 initial guess `x0` using finite-difference Levenberg-Marquardt iterations. Scalar guesses return scalars, and vector/matrix guesses return solutions with the same shape as `x0`.",

    keywords: &["fsolve",
"nonlinear solve",
"root finding",
"Jacobian",
"Levenberg-Marquardt"],

    related: &["fminunc",
"fzero",
"optimset",
"linsolve"],

    sections: &[BuiltinDocumentationSection {
 heading: "Behavior and supported forms",
 paragraphs: &["The function handle must return a finite real scalar or vector.",
"Strict compatibility mode rejects typed-integer or logical initial guesses and residuals. RunMat extension mode accepts those values when every integer converts exactly to binary64.",
"Typed-integer option controls are a separately gated RunMat extension. Structural counts such as `MaxIter` and `MaxFunEvals` are validated as exact nonnegative integers rather than rounded through floating point.",
"Jacobians are estimated with forward finite differences.",
"The step is chosen with a Levenberg-Marquardt damping parameter and accepted only when it reduces the residual norm.",
"Options are supplied as a struct, usually created with `optimset`. `TolX`, `TolFun`, `MaxIter`, `MaxFunEvals`, and `Display` are accepted.",
"Up to five outputs are supported: `x`, `fval`, `exitflag`, `output`, and the final Jacobian. A stationary non-root or exhausted budget is returned with a nonpositive convergence status instead of being raised as an evaluation error.",
"The initial guess and stopping tolerances affect convergence. Inspect the residual and convergence status to distinguish a root from a stationary non-root."] }
,

BuiltinDocumentationSection {
 heading: "Execution and residency",
 paragraphs: &["The nonlinear solver runs on the host and does not provide native gpuArray or tall-array execution. In RunMat extension mode, resident numeric operands and callback results use a separately gated gather fallback. Strict compatibility mode rejects that extension before provider access.",
"The solver state uses host binary64; there is no provider-native solver kernel or numeric fusion producer. Callback computations retain their own capabilities, effects and acceleration policies. Native and browser routes use the same algorithm and the owning execution context's cooperative CPU budget.",
"Cancellation is checked between bounded input/result scans and solver steps. Native dense linear algebra calls, where used, run to completion between checks rather than being hard-preempted."] }
,

BuiltinDocumentationSection {
 heading: "Example notes",
 paragraphs: &["The two-equation system has more than one root. The example checks the residual rather than requiring a particular ordering of the solution coordinates."] }
],

    examples: &[BuiltinExample {
 id: "example-1",
 title: "Solve a coupled nonlinear system",
 program: "F = @(x) [x(1)^2 + x(2)^2 - 4; x(1)*x(2) - 1];\nx = fsolve(F, [1; 1])",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(isequal(size(x),[2 1])); assert(all(isfinite(x))); assert(norm(F(x))<1e-5);" }
 }
,

BuiltinExample {
 id: "example-2",
 title: "Solve a scalar equation",
 program: "x = fsolve(@sin, 3)",
 display_output: Some("x =\n    3.1416"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(abs(x-pi)<1e-5); assert(abs(sin(x))<1e-5);" }
 }
],

    example_exemption: None,

    faqs: &[],

    links: &[BuiltinDocumentationLink {
 label: "fminunc",
 target: BuiltinDocumentationLinkTarget::Builtin("fminunc") }
,

BuiltinDocumentationLink {
 label: "fzero",
 target: BuiltinDocumentationLinkTarget::Builtin("fzero") }
,

BuiltinDocumentationLink {
 label: "optimset",
 target: BuiltinDocumentationLinkTarget::Builtin("optimset") }
,

BuiltinDocumentationLink {
 label: "linsolve",
 target: BuiltinDocumentationLinkTarget::Builtin("linsolve") }
,

BuiltinDocumentationLink {
 label: "Implementation",
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/fsolve.rs") }
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
 location: "builtins::math::optim::fsolve::tests" }
,
BuiltinEvidenceReference {
 kind: BuiltinEvidenceKind::IntegrationTest,
 label: "Retained integration source",
 location: "runmat-vm/tests/closures.rs::fsolve_accepts_anonymous_vector_function" }
],

        notes: &["Source references are not execution receipts. Native, browser, callback, residency and standalone examples require integrated qualification."],

    }
,

    introduced: None,

    status: Some(BuiltinDocumentationStatus::Partial),

}
;
