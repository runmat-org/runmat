use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {

    authority: BuiltinDocumentationAuthority::Catalog,

    title: Some("lsqnonlin"),
 slug: Some("lsqnonlin"),

    summary: "Solve bounded and unbounded nonlinear least-squares residual problems.",

    description: "`lsqnonlin` minimizes the squared 2-norm of a vector-valued residual callback `fun(x)` using a bound-reflective Gauss-Newton trust-region controller or Levenberg-Marquardt. Scalar, vector, and matrix `x0` inputs are supported and the returned solution preserves the shape of `x0`.",

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
 paragraphs: &["FiniteDifferenceStepSize (legacy FinDiffRelStep) accepts a finite strictly positive real floating scalar or vector; a scalar factor expands to every variable. TypicalX must be a finite strictly positive real floating vector with one element per original variable (a scalar for a scalar problem).",
concat!(
"Omitted or empty controls resolve to sqrt(eps) for forward differences, eps^(1/3) for existing central-difference routes, and unit TypicalX. ",
"Forward offsets use sign(x), with sign(0)=1, times the factor and max(abs(x),TypicalX); central offsets omit that sign. Actual representable ",
"bound-feasible offsets own the denominators. Nominal underflow uses adjacent representable feasible points; unusable actual offsets or an ",
"overflowing central span refuse before probe callbacks. These controls do not add objective calls or bypass existing evaluation budgets. ",
"Fully supplied derivative paths leave these operands dormant, without validating or gathering them. This least-squares implementation ",
"remains forward-difference only; FiniteDifferenceType='central' remains rejected, not accepted and ignored."
),
"The residual callback is evaluated as `fun(x)` and must return a finite real numeric scalar, vector, or array. The residual output preserves the callback's initial shape.",
"Lower and upper bounds may be empty or contain up to `numel(x0)` elements. Supplied bounds apply to the leading variables; omitted trailing entries are unbounded, matching MATLAB's partial-bound convention.",
"`optimset` and `optimoptions` structs are accepted, and `[]` option placeholders are treated as defaults. `StepTolerance` (`TolX`), `FunctionTolerance`, `OptimalityTolerance`, `MaxIterations` (`MaxIter`), `MaxFunctionEvaluations` (`MaxFunEvals`), `Display`, and `Algorithm` are recognized. The legacy `TolFun` control supplies function and optimality defaults; conflicting explicit legacy and modern controls are rejected.",
concat!(
"Algorithm='trust-region-reflective' selects a bound-reflective Gauss-Newton trust region. The local residual Jacobian and Coleman-Li ",
"distance scaling define its model; the factorization step uses rectangular QR/COD with diagonal regularization and a secular trust-radius ",
"solve, without constructing normal equations. A reflected feasible proposal is evaluated against the original residual, and its ",
"actual/predicted reduction controls acceptance and radius changes. Algorithm='levenberg-marquardt' selects the separate damping controller."
),
"Reflective requires at least as many residual entries as variables. Otherwise the actual method is Levenberg-Marquardt, with a warning, and the already evaluated starting residual is reused without enlarging the selected evaluation budget. output.algorithm reports the method actually used. Requested reflective starts on nonfixed bounds move to an adjacent representable interior point before the first callback. Fixed coordinates remain fixed and do not consume derivative probes.",
concat!(
"Display='iter' and 'final' write iteration or final summaries through the original console/diary services; off and none are silent. The ",
"effective default evaluation limit resolves from the variable count (100 times the number of variables for reflective, 200 for ",
"Levenberg-Marquardt), and zero iterations are honored. Iteration and function limits are nonnegative integers; step, function and optimality ",
"tolerances are nonnegative finite scalars. Reflective optimality uses OptimalityTolerance; Levenberg-Marquardt uses an internal optimality ",
"tolerance of 1e-4 times FunctionTolerance and leaves OptimalityTolerance dormant."
),
concat!(
"Reflective SubproblemAlgorithm='factorization' uses rectangular QR/COD and a secular trust-radius solve. SubproblemAlgorithm='cg' uses ",
"matrix-free Steihaug CG over the original rectangular Jacobian. PrecondBandWidth selects the retained preconditioner band (zero is ",
"diagonal); Inf selects factorization. MaxPCGIter defaults to max(1,floor(numel(x0)/2)) and TolPCG defaults to 0.1; both must be positive. ",
"Only the effective reflective method inspects these controls; the underdetermined Levenberg-Marquardt fallback leaves them dormant. ",
"JacobPattern, central differences, output/plot callbacks and parallel derivatives are not admitted by this profile. ",
"Model and callback errors, including nonfinite numerical residual values, propagate; they are not caught and replayed as another method."
),
concat!(
"With SpecifyObjectiveGradient=true (legacy Jacobian='on'), fun returns [F,J] in one callback invocation. J is a finite real single or double ",
"m-by-n dense or sparse matrix. The accepted derivative remains paired with its original point; rejected trials cannot replace it. Reflective ",
"products borrow this backing directly. Levenberg-Marquardt materializes its required dense numerical work, while the returned supplied ",
"Jacobian preserves its original dense/sparse storage and numeric class. No finite-difference callbacks or finite-difference control gathers ",
"occur on this supplied path."
),
concat!(
"With SpecifyObjectiveGradient=true and JacobianMultiplyFcn=jmfun (legacy JacobMult), fun returns [F,Jinfo]. Jinfo can be any supported ",
"data type and remains rooted with its accepted point. jmfun(Jinfo,Y,flag) returns finite real single or double products: flag +1 uses ",
"n-by-1 Y and returns J*Y of size m-by-1; flag -1 uses m-by-1 Y and returns J'*Y of size n-by-1; flag 0 uses n-by-2 Y and returns J'*(J*Y) ",
"of size n-by-2. Here m=numel(F) and n=numel(x0). A one-direction Krylov request uses an exact zero second column in the normal-product pair."
),
concat!(
"CG keeps the Jacobian implicit and retains only the requested preconditioner band. RunMat's selected direct factorization forms its ",
"required rectangular columns through forward products. If an underdetermined ",
"reflective request falls back to Levenberg-Marquardt, its required dense numerical matrix is likewise formed from the same accepted Jinfo. ",
"Neither boundary replays the objective. Explicit Levenberg-Marquardt leaves JacobianMultiplyFcn dormant and expects a numeric supplied J."
),
concat!(
"Requested output seven is an actual m-by-n matrix at the accepted terminal point, not Jinfo. It is formed through forward columns only ",
"when needed, or reuses already available accepted LM work. It never uses rejected or pre-step information. Objective function counts ",
"count calls to fun, including derivative probes, not jmfun products; multiply callbacks use the same original cooperative CPU/cancellation ",
"owner and their work is bounded by the selected solver/PCG controls. Callback failures propagate without trying another method."
),
"MaxFunctionEvaluations=0 stops before any residual callback. RunMat returns the bound-projected supplied point with its original shape, flag 0, and zero iterations and function count. Residual, Jacobian and multipliers are empty; resnorm and firstorderopt are NaN. output.algorithm is not-started and output.requestedalgorithm records the requested selector. This RunMat stopped-before-evaluation convention does not assert feasibility, a root or stationarity.",
"Reflective firstorderopt is the bound-distance-scaled free-coordinate measure. Derivatives in fixed coordinates cannot be estimated without leaving the domain: their final Jacobian entries are NaN, and bound multiplier arrays are empty when derivative evidence is incomplete for either method. This projected stopping measure does not assert a known full gradient.",
"Finite host-memory policies are refused before compiler, Jacobian or factorization workspace allocation. The solver does not claim provider-native or distributed nonlinear least-squares execution.",
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
 paragraphs: &["`lsqnonlin` gathers automatically resident residuals, bounds and supplied Jacobians through the original context's canonical host boundary. Explicit gpuArray inputs require the existing separately admitted intent extension; this profile does not grant one. Callback bodies can still use GPU-aware builtins; solver iterations and factorizations use host arithmetic.",
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
 title: "Supply a Jacobian and select CG",
 program: "opts = optimoptions(\"lsqnonlin\", \"SpecifyObjectiveGradient\", true, \"SubproblemAlgorithm\", \"cg\", \"PrecondBandWidth\", 0, \"MaxPCGIter\", 20, \"Display\", \"off\");\nr = @(x) deal([x(1)-1; 2*x(2)-4], single([1 0; 0 2]));\n[x, resnorm, residual, exitflag, output, lambda, jacobian] = lsqnonlin(r, [0; 0], [], [], opts)",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(norm(x-[1;2])<1e-4); assert(resnorm<1e-8); assert(isequal(size(jacobian),[2 2])); assert(isa(jacobian,\"single\")); assert(output.cgiterations>0); assert(isstruct(lambda));" }
 }
,
BuiltinExample {
    id: "example-4",
    title: "Use structured Jacobian information",
    program: concat!(
        "opts=optimoptions('lsqnonlin','SpecifyObjectiveGradient',true,",
        "'JacobianMultiplyFcn',@structuredProduct,'SubproblemAlgorithm','cg',",
        "'PrecondBandWidth',0,'Display','off');\n",
        "r=@(x) deal([x(1)-1;2*x(2)-4],struct('A',[1 0;0 2]));\n",
        "[x,rn,~,~,out,~,J]=lsqnonlin(r,[0;0],[],[],opts);\n",
        "function W=structuredProduct(info,Y,flag)\n",
        "if flag>0\nW=info.A*Y;\n",
        "elseif flag<0\nW=info.A'*Y;\n",
        "else\nW=info.A'*(info.A*Y);\nend\nend"
    ),
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(norm(x-[1;2])<1e-4); assert(rn<1e-8); assert(norm(J-[1 0;0 2])<1e-12); assert(out.cgiterations>0);"
    }
}
],

    example_exemption: None,

    faqs: &[BuiltinDocumentationFaq {
 question: "Does lsqnonlin support linear or nonlinear constraints?",
 answer: "Only unconstrained and bound-constrained residual problems are supported. Empty A, b, Aineq, bineq, Aeq, beq and nonlcon placeholders are accepted; nonempty constraints are rejected." }
,

BuiltinDocumentationFaq {
 question: "Does lsqnonlin run on the GPU?",
 answer: "The solver is host iterative work. Callback code may call GPU-aware builtins, but residuals and bounds are gathered for the solver. Provider-native nonlinear least-squares execution is unavailable." }
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
