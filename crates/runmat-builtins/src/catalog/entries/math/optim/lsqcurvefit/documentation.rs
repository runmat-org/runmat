use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {

    authority: BuiltinDocumentationAuthority::Catalog,

    title: Some("lsqcurvefit"),
 slug: Some("lsqcurvefit"),

    summary: "Fit nonlinear parametric curves in the least-squares sense.",

    description: "`x = lsqcurvefit(fun, x0, xdata, ydata)` estimates parameters `x` so that `fun(x, xdata)` matches `ydata` in the least-squares sense. RunMat preserves the shape of `x0`, supports scalar and vector parameters, accepts lower and upper parameter bounds, and can return solver diagnostics through MATLAB-compatible multi-output forms.",

    keywords: &["lsqcurvefit",
"least squares",
"curve fitting",
"optimization",
"Levenberg-Marquardt",
"bounds"],

    related: &["fsolve",
"optimoptions",
"optimset"],

    sections: &[BuiltinDocumentationSection {
 heading: "Behavior and supported forms",
 paragraphs: &["FiniteDifferenceStepSize (legacy FinDiffRelStep) accepts a finite strictly positive real floating scalar or vector; a scalar factor expands to every variable. TypicalX must be a finite strictly positive real floating vector with one element per original variable (a scalar for a scalar problem).",
"Omitted or empty controls resolve to sqrt(eps) for forward differences, eps^(1/3) for existing central-difference routes, and unit TypicalX. Forward offsets use sign(x), with sign(0)=1, times the factor and max(abs(x),TypicalX); central offsets omit that sign. Actual representable bound-feasible offsets own the denominators. Nominal underflow uses adjacent representable feasible points; unusable actual offsets or an overflowing central span refuse before probe callbacks. These controls do not add objective calls or bypass existing evaluation budgets. Fully supplied derivative paths leave these operands dormant, without validating or gathering them. This least-squares implementation remains forward-difference only; FiniteDifferenceType='central' remains rejected, not accepted and ignored.",
"The model callback is evaluated as `fun(x, xdata)` and must return a finite real scalar or array with the same shape as `ydata`.",
"Unbounded problems use finite-difference Levenberg-Marquardt iterations.",
"Bounded problems project trial points into `[lb, ub]` and estimate finite-difference Jacobians without stepping outside finite bounds.",
"`lb` and `ub` may be empty, scalar, or arrays with the same number of elements as `x0`; scalar bounds are broadcast to every parameter.",
"`optimset` and `optimoptions` option structs are accepted. `TolX`, `TolFun`, `MaxIter`, `MaxFunEvals` (also `MaxFunctionEvaluations` when constructing options), `Display`, and `Algorithm` are recognized.",
"The documented numeric solver domain is floating point. Native integer x0, xdata, ydata, bounds, numeric options, and callback results are independently gated RunMat extensions.",
"Integer x0, ydata, bounds, floating options, and callback results must be exactly representable at the double solver boundary. Integer xdata remains an exact callback payload and is not eagerly converted.",
"`Algorithm` accepts `levenberg-marquardt` and `trust-region-reflective`; both currently use RunMat's projected Levenberg-Marquardt implementation.",
"With requested outputs, RunMat supports `[x, resnorm, residual, exitflag, output, lambda, jacobian]`."] }
,

BuiltinDocumentationSection {
 heading: "Execution and residency",
 paragraphs: &["The iterative nonlinear least-squares solver runs on the host. Automatic residency gathers transparently at host boundaries; explicit gpuArray host fallback is a gated RunMat extension. Exact xdata payloads are passed to callbacks unchanged.",
"The solver state uses host binary64; there is no provider-native solver kernel or numeric fusion producer. Callback computations retain their own capabilities, effects and acceleration policies. Native and browser routes use the same algorithm and the owning execution context's cooperative CPU budget.",
"Cancellation is checked between bounded input/result scans and solver steps. Native dense linear algebra calls, where used, run to completion between checks rather than being hard-preempted."] }
],

    examples: &[BuiltinExample {
 id: "example-1",
 title: "Fit a line",
 program: "xdata = [0 1 2 3];\nydata = 2.*xdata + 1;\np = lsqcurvefit(@(p,x) p(1).*x + p(2), [0; 0], xdata, ydata)",
 display_output: Some("p =\n    2.0000\n    1.0000"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(isequal(size(p),[2 1])); assert(norm(p-[2;1])<1e-4);" }
 }
,

BuiltinExample {
 id: "example-2",
 title: "Fit an exponential decay",
 program: "xdata = 0:0.2:2;\nydata = 2.5.*exp(-0.7.*xdata);\np = lsqcurvefit(@(p,x) p(1).*exp(-p(2).*x), [1; 0.1], xdata, ydata)",
 display_output: Some("p =\n    2.5000\n    0.7000"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(norm(p-[2.5;0.7])<1e-3);" }
 }
,

BuiltinExample {
 id: "example-3",
 title: "Fit with bounds and diagnostics",
 program: "opts = optimoptions(\"lsqcurvefit\", \"TolX\", 1e-8, \"Display\", \"off\");\n[p, resnorm, residual, exitflag, output] = lsqcurvefit(@(p,x) p(1).*x, 0.5, [1 2 3], [2 4 6], 0, 1, opts)",
 display_output: Some("p =\n    1.0000"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(abs(p-1)<1e-5); assert(abs(resnorm-14)<1e-3); assert(isequal(size(residual),[1 3])); assert(isstruct(output));" }
 }
],

    example_exemption: None,

    faqs: &[BuiltinDocumentationFaq {
 question: "Does lsqcurvefit run on the GPU?",
 answer: "The solver itself is host-side because it repeatedly invokes user callbacks and adapts finite-difference steps. The callback can still use GPU-aware builtins, but the residual vector is gathered before each solver step." }
,

BuiltinDocumentationFaq {
 question: "What is returned in lambda?",
 answer: "`lambda.lower` and `lambda.upper` are approximate bound multipliers computed from the final finite-difference Jacobian and residual." }
,

BuiltinDocumentationFaq {
 question: "Are nonlinear constraints supported?",
 answer: "RunMat currently supports the core curve-fitting forms with optional lower and upper parameter bounds. Linear-constraint, nonlinear-constraint, and problem-structure forms are not yet implemented." }
],

    links: &[BuiltinDocumentationLink {
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
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/lsqcurvefit.rs") }
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
 location: "builtins::math::optim::lsqcurvefit::tests" }
,
BuiltinEvidenceReference {
 kind: BuiltinEvidenceKind::IntegrationTest,
 label: "Retained integration source",
 location: "runmat-vm/tests/closures.rs::lsqcurvefit_accepts_anonymous_curve_model" }
],

        notes: &["Source references are not execution receipts. Native, browser, callback, residency and standalone examples require integrated qualification."],

    }
,

    introduced: None,

    status: Some(BuiltinDocumentationStatus::Partial),

}
;
