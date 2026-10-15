use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {

    authority: BuiltinDocumentationAuthority::Catalog,

    title: Some("fminunc"),
 slug: Some("fminunc"),

    summary: "Find unconstrained local minima in MATLAB and RunMat.",

    description: "`x = fminunc(fun, x0)` minimizes a finite real scalar objective starting from binary64 `x0`. Scalar guesses return scalars, and vector or array guesses return solutions with the same shape as `x0`.",

    keywords: &["fminunc",
"unconstrained minimization",
"optimization",
"BFGS",
"quasi-Newton",
"line search"],

    related: &["fminbnd",
"fsolve",
"optimset"],

    sections: &[BuiltinDocumentationSection {
 heading: "Behavior and supported forms",
 paragraphs: &["Uses a host-side BFGS quasi-Newton method with a strong-Wolfe line search.",
"Strict compatibility mode rejects typed-integer or logical initial guesses, objective values, and supplied gradients. RunMat extension mode accepts those values when every integer converts exactly to binary64.",
"Typed-integer option controls are a separately gated RunMat extension. Structural counts such as `MaxIter` and `MaxFunEvals` are validated as exact nonnegative integers rather than rounded through floating point.",
"When `SpecifyObjectiveGradient` is false or omitted, gradients are estimated with forward finite differences.",
"When `SpecifyObjectiveGradient` is true, the objective callback is called as `[f, g] = fun(x)` and `g` must contain one gradient entry per element of `x`.",
"Options are supplied as a struct, usually from `optimoptions` or `optimset`. `TolX`, `TolFun`, `MaxIter`, `MaxFunEvals`, `Display`, `Algorithm`, and `SpecifyObjectiveGradient` are accepted.",
"Supported `Algorithm` values are `'quasi-newton'` and `'bfgs'`; trust-region mode is not enabled in this implementation.",
"Multi-output forms return `x`, `fval`, `exitflag`, `output`, `grad`, and an approximate Hessian."] }
,

BuiltinDocumentationSection {
 heading: "Execution and residency",
 paragraphs: &["The BFGS optimization loop runs on the host and does not provide native gpuArray or tall-array execution. In RunMat extension mode, resident numeric operands, objective values, and gradients use a separately gated gather fallback. Strict compatibility mode rejects that extension before provider access.",
"`fminunc` is a host-side optimization sink. GPU-resident initial guesses are gathered before the solve, and objective callbacks may opt back into GPU work internally.",
"The canonical host-materialization boundary is required because BFGS iterations need scalar objective values, gradient vectors, and adaptive line-search decisions on the CPU.",
"Providers do not implement a direct `fminunc` hook; acceleration remains available inside user callback computations.",
"The solver state uses host binary64; there is no provider-native solver kernel or numeric fusion producer. Callback computations retain their own capabilities, effects and acceleration policies. Native and browser routes use the same algorithm and the owning execution context's cooperative CPU budget.",
"Cancellation is checked between bounded input/result scans and solver steps. Native dense linear algebra calls, where used, run to completion between checks rather than being hard-preempted."] }
],

    examples: &[BuiltinExample {
 id: "example-1",
 title: "Minimize a quadratic vector objective",
 program: "fun = @(x) sum((x - [1; 2; 3]).^2);\nx = fminunc(fun, [0; 0; 0])",
 display_output: Some("x =\n    1.0000\n    2.0000\n    3.0000"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(isequal(size(x),[3 1])); assert(norm(x-[1;2;3])<1e-4);" }
 }
,

BuiltinExample {
 id: "example-2",
 title: "Use an objective gradient",
 program: "fun = @(x) deal(sum((x - [1; 2]).^2), 2*(x - [1; 2]));\nopts = optimoptions('fminunc', 'SpecifyObjectiveGradient', true);\n[x, fval, exitflag, output, grad] = fminunc(fun, [0; 0], opts)",
 display_output: Some("x =\n    1.0000\n    2.0000\n\nfval = 0\nexitflag = 1"),
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(norm(x-[1;2])<1e-4); assert(abs(fval)<1e-8); assert(norm(grad)<1e-4); assert(isstruct(output));" }
 }
],

    example_exemption: None,

    faqs: &[BuiltinDocumentationFaq {
 question: "Does fminunc use the GPU?",
 answer: "The optimizer state and line search run on the host because the solver repeatedly invokes user callbacks. The callback itself can use GPU-aware operations." }
,

BuiltinDocumentationFaq {
 question: "What is in the output struct?",
 answer: "`output` contains `iterations`, `funcCount`, `algorithm`, `firstorderopt`, `stepsize`, and `message` fields." }
],

    links: &[BuiltinDocumentationLink {
 label: "fminbnd",
 target: BuiltinDocumentationLinkTarget::Builtin("fminbnd") }
,

BuiltinDocumentationLink {
 label: "fsolve",
 target: BuiltinDocumentationLinkTarget::Builtin("fsolve") }
,

BuiltinDocumentationLink {
 label: "optimset",
 target: BuiltinDocumentationLinkTarget::Builtin("optimset") }
,

BuiltinDocumentationLink {
 label: "Implementation",
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/fminunc.rs") }
],

    media: &[],

    evidence: BuiltinDocumentationEvidence {

        implementation: &[
            BuiltinDocumentationLink {
 label: "Solver callback and numeric preparation",
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/common.rs") }
,


        ],

        verification: &[BuiltinEvidenceReference {
 kind: BuiltinEvidenceKind::UnitTest,
 label: "Retained unit source",
 location: "builtins::math::optim::fminunc::tests" }
,
BuiltinEvidenceReference {
 kind: BuiltinEvidenceKind::UnitTest,
 label: "Retained options source",
 location: "builtins::math::optim::optimoptions::tests::optimoptions_fminunc_defaults_match_solver" }
],

        notes: &["Source references are not execution receipts. Native, browser, callback, residency and standalone examples require integrated qualification."],

    }
,

    introduced: None,

    status: Some(BuiltinDocumentationStatus::Partial),

}
;
