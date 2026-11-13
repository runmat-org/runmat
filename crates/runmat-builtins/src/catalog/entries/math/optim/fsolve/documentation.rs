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
 paragraphs: &["FiniteDifferenceStepSize (legacy FinDiffRelStep) accepts a finite strictly positive real floating scalar or vector; a scalar factor expands to every variable. TypicalX must be a finite strictly positive real floating vector with one element per original variable (a scalar for a scalar problem).",
"Omitted or empty controls resolve to sqrt(eps) for forward differences, eps^(1/3) for existing central-difference routes, and unit TypicalX. Forward offsets use sign(x), with sign(0)=1, times the factor and max(abs(x),TypicalX); central offsets omit that sign. Actual representable bound-feasible offsets own the denominators. Nominal underflow uses adjacent representable feasible points; unusable actual offsets or an overflowing central span refuse before probe callbacks. These controls do not add objective calls or bypass existing evaluation budgets. Fully supplied derivative paths leave these operands dormant, without validating or gathering them. This least-squares implementation remains forward-difference only; FiniteDifferenceType='central' remains rejected, not accepted and ignored.",
"A requested fifth output is the Jacobian at the returned iterate. With MaxIterations=0, finite differences are evaluated only when that output is requested and the hard MaxFunEvals budget permits. Paired analytic Jacobians require no additional callback. If the budget leaves final coefficients unevaluated, those entries are NaN (fixed-bound columns remain zero); an unmeasured firstorderopt is NaN. Unevaluated or previous-iterate zeros are not published as measured derivatives.",
"SpecifyObjectiveGradient=true requires [F,J] from the objective. J may be real dense or sparse CSC, with one row per residual element and one column per variable. Sparse storage is validated and converted at the existing dense LM workspace boundary; this is not a sparse LM solver kernel. Each paired evaluation counts once; the supplied Jacobian is paired with the accepted iterate and moved into the final result, including iteration/evaluation-limit exits, rather than recomputed through finite differences or taken from a rejected trial.",
"Finite host-memory execution is unsupported: requests with a finite host-memory policy fail before objective callbacks or solver workspace allocation; ordinary unbounded execution is unchanged.",
"The function handle must return a finite real scalar or vector.",
"Strict compatibility mode rejects typed-integer or logical initial guesses and residuals. RunMat extension mode accepts those values when every integer converts exactly to binary64.",
"Typed-integer option controls are a separately gated RunMat extension. Structural counts are read exactly rather than rounded through floating point. Iteration limits are nonnegative; function-evaluation limits must be positive.",
"Jacobians use forward finite differences by default, or the paired objective Jacobian when SpecifyObjectiveGradient is true.",
"The step is chosen with a Levenberg-Marquardt damping parameter and accepted only when it reduces the residual norm.",
"Options are supplied as a struct, usually created with `optimset`. `TolX`, `TolFun`, `MaxIter` (`MaxIterations`), `MaxFunEvals`, `Display`, and `SpecifyObjectiveGradient` are accepted.",
"Up to five outputs are supported: `x`, `fval`, `exitflag`, `output`, and the final Jacobian. A stationary non-root or exhausted budget is returned with a nonpositive convergence status instead of being raised as an evaluation error.",
"The initial guess and stopping tolerances affect convergence. Inspect the residual and convergence status to distinguish a root from a stationary non-root."] }
,

BuiltinDocumentationSection {
 heading: "Execution and residency",
 paragraphs: &["The solver runs on the host and has no provider-native solver kernel or tall-array execution. Authenticated automatic placement can be gathered transparently under either language compatibility mode. Explicit GPU operands and callback results require the RunMat resident-fallback extension; strict MATLAB mode rejects them before provider access. Device operands with unproven placement require an explicit gather. Compatibility, cancellation and provider authority come from the original invocation.",
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
