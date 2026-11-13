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
"line search", "trust-region", "Hessian", "HessianMultiplyFcn"],

    related: &["fminbnd",
"fsolve",
"optimset"],

    sections: &[BuiltinDocumentationSection {
 heading: "Behavior and supported forms",
 paragraphs: &["FiniteDifferenceStepSize (legacy FinDiffRelStep) accepts a finite strictly positive real floating scalar or vector; a scalar factor expands to every variable. TypicalX must be a finite strictly positive real floating vector with one element per original variable (a scalar for a scalar problem).",
"Omitted or empty controls resolve to sqrt(eps) for forward differences, eps^(1/3) for existing central-difference routes, and unit TypicalX. Forward offsets use sign(x), with sign(0)=1, times the factor and max(abs(x),TypicalX); central offsets omit that sign. Actual representable bound-feasible offsets own the denominators. Nominal underflow uses adjacent representable feasible points; unusable actual offsets or an overflowing central span refuse before probe callbacks. These controls do not add objective calls or bypass existing evaluation budgets. Fully supplied derivative paths leave these operands dormant, without validating or gathering them. fminunc trust-region does not consume either control; its supplied-gradient Hessian machinery is unchanged.",
"OutputFcn accepts a function handle, a cell array of handles, or []. Observers receive (x, optimValues, state). For quasi-Newton, init precedes iteration, iter follows accepted updates, and done marks normal completion or requested stop. Its optimValues includes iteration, funccount, fval, gradient, firstorderopt, stepsize, searchdirection, lssteplength, directionalderivative, and a blank procedure field. Direction/line-search quantities are empty before a step exists; x retains the original input shape. A logical true return requests exitflag -1 without additional objective evaluations. Observer errors and cancellation propagate without replay.",
"Both algorithms reject finite host-memory policies before callbacks or optimizer workspace allocation.",
"The default host algorithm is BFGS quasi-Newton with a strong-Wolfe line search. Modern HessianApproximation selects bfgs, lbfgs (history 10), or {'lbfgs',m} with a positive integer history. Legacy optimset HessUpdate also selects genuine DFP or steepdesc. L-BFGS uses retained accepted secant pairs and two-loop recursion; steepdesc uses the negative gradient without an inverse model. BFGS/DFP retain the current dense 2048-variable limit; L-BFGS/steepest do not allocate that dense matrix or inherit its dimension limit. Conflicting supplied modern/legacy update controls reject. Algorithm='trust-region' with SpecifyObjectiveGradient=true instead uses a quadratic trust model, ratio-based trial acceptance and radius updates, and truncated preconditioned CG or an explicit self-adjoint factorization/secular subproblem solve. These are RunMat numerical implementations, not a claim of byte-identical MATLAB iterates, subspace choices or console presentation.",
"Strict compatibility mode rejects typed-integer or logical initial guesses, objective values, and supplied gradients. RunMat extension mode accepts those values when every integer converts exactly to binary64.",
"Typed-integer option controls are a separately gated RunMat extension. Structural counts are read exactly rather than rounded through floating point. Iteration limits are nonnegative; function-evaluation limits must be positive.",
"When `SpecifyObjectiveGradient` is false or omitted, gradients initially use forward finite differences. In RunMat's adaptive numerical policy, a numerical line-search failure can trigger one one-way refinement to central differences at the current accepted point, retaining its existing objective value. The refinement requires two actual objective calls per variable within the original evaluation budget, then future complete states cost one objective plus two calls per variable. Insufficient budget stops with flag 0; callback errors and cancellation are never caught and retried. Analytic gradients are not refined. This precision safeguard is not a claim of MATLAB's identical finite-difference or line-search implementation.",
"When `SpecifyObjectiveGradient` is true, the objective callback is called as `[f, g] = fun(x)` and `g` must contain one gradient entry per element of `x`.",
"Options are supplied as a struct, usually from `optimoptions` or `optimset`. `TolX`, `TolFun`, `MaxIter`, `MaxFunEvals`, `Display`, `Algorithm`, `SpecifyObjectiveGradient`, and `OutputFcn` are accepted; `MaxIterations` aliases `MaxIter`. For effective quasi-Newton, legacy TolFun means OptimalityTolerance; for effective trust-region it means FunctionTolerance. Modern OptimalityTolerance and FunctionTolerance are distinct controls. Nonnegative step, function and optimality tolerances may be zero. Explicitly conflicting legacy/modern bindings reject.",
"Supported Algorithm values are 'quasi-newton', 'bfgs' and 'trust-region'. Requested trust-region without a supplied gradient emits a warning and falls back to quasi-Newton. Trust-region-only settings are then inactive: their payloads are not evaluated, transferred or invoked.",
"Multi-output forms return x, fval, exitflag, output, grad, and the algorithm-specific sixth model/Hinfo output.",
"L-BFGS returns [] for its sixth output and performs no final Hessian probes. For BFGS, DFP and steepest descent, a requested sixth output is a final finite-difference Hessian estimate, not the inverse-update approximation or a fabricated identity. It reuses the accepted gradient and differences actual gradients at represented displacements. Analytic gradients cost n extra objective/gradient calls; numerical gradients cost n complete gradient states under the current derivative mode. The complete sweep is admitted within the original remaining MaxFunctionEvaluations before allocation or any partial sweep, and funcCount/done include actual probes. Unrequested Hessians perform no probes. For these estimated-Hessian forms, insufficient remaining budget or an ObjectiveLimit stop that precludes extra probes raises RunMat:fminunc:HessianUnavailable; this is an explicit finite-budget compatibility frontier, not full MATLAB sixth-output parity.",
"UseParallel='off' (or false) selects serial numerical differences. 'on' requires an eligible selected native-process or browser-worker pool and creates it using that execution adapter's existing configuration when none is open. 'auto' (or true) uses an already-open eligible pool; without one it takes the serial route. RunMat does not invent an automatic pool-creation preference.",
"Forward/central differences, numerical refinement and the numerical-gradient sweeps used by a requested final Hessian submit bounded ordered worker waves under the original selected callback/source authority. The complete difference-call budget is admitted before a sweep; f0 is retained, and callback/provider/cancellation failures never replay serially.",
"A browser host must provide selectedProgramWork, isolated Direct/Graph workers, their actual module/factory workerInventory, and launchProgram/cancelProgram methods. The inventory declares supported host services and worker-local devices; stock CPU workers do not borrow the UI's network, filesystem, graphics, foreign services or GPU. These methods return Runtime-owned invocation-retirement terminals tied to the complete original assignment.",
"Invocation input/source roots, child work, and untransferred resources must retire before publication; matching task IDs or posting cancellation alone does not prove retirement.",
"Rejected, malformed, or unverified terminals retain the original source owners and resource reservation. The ownedBuiltinWork flag and its literal physical-exit cancelOwned protocol do not authorize selected-program execution. Without qualified program-worker authority, on refuses and auto uses serial execution.",
"Supplied objective/gradient states remain on the calling lane. With a supplied gradient this control is inactive and is not gathered or interpreted. Update controls likewise remain untouched under effective trust-region. L-BFGS history lengths inside a cell specification must be host scalar controls."] }
,

BuiltinDocumentationSection {
 heading: "Trust-region models and controls",
 paragraphs: &["Trust-region requires the objective gradient. With HessianFcn='objective', fun is invoked as [f,g,H]=fun(x); H must be a finite real symmetric n-by-n dense or sparse matrix. Otherwise an active HessianMultiplyFcn uses [f,g,Hinfo]=fun(x) and W=hmfun(Hinfo,Y), where Y and W have n rows. Hinfo remains its original rooted carrier, without conversion or a metadata clone. Objective Hessians take precedence over multiply/pattern controls; active multiply takes precedence over HessPattern. Inactive controls are not evaluated.",
"Without a provided matrix or multiply function, the model differences supplied gradients using represented forward displacements. A finite symmetric real/logical HessPattern supplies structural nonzeros; column coloring reduces the actual gradient callback count and yields a sparse model. User-supplied structural zeros and Hessian products are mathematical promises, not independently proved derivatives. Model construction reserves its complete callback sweep plus a trial within the original MaxFunctionEvaluations budget. No callback is replayed to fill a sixth output.",
"SubproblemAlgorithm='cg' computes a truncated preconditioned conjugate-gradient step inside the trust region, with a boundary step when curvature is negative. MaxPCGIter is positive and defaults to max(1,floor(n/2)); TolPCG is positive and defaults to .1. PrecondBandWidth is a nonnegative exact integer, default zero, selecting a band preconditioner. Its symmetric band receives magnitude-aware Gershgorin diagonal boosts where needed for positive definiteness, then is factored, while products and predicted reduction use the original Hessian.",
"For an opaque Hinfo model, exact band construction requires n basis-product callbacks once per constructed accepted-state model; it is reused across rejected trials. Each PCG iteration adds one Hessian product, with a further product for predicted reduction. Explicit dense/CSC matrices supply their band directly, without n redundant matvecs. Default diagonal matrix-free live storage is linear in n; an explicitly requested band requires O(n*(band+1)) preconditioner storage and can approach dense storage. The matrix-free CG path does not form a separate dense Hessian or inverse, and does not cache across model generations based on Hinfo equality.",
"SubproblemAlgorithm='factorization', or floating PrecondBandWidth=Inf under cg, selects an explicit self-adjoint EVD/secular trust subproblem with an indefinite hard-case treatment. When explicitly selected with HessianMultiplyFcn, direct mode assembles n basis products into its owned dense numerical model, preserving the original Hinfo output; this has n-squared direct workspace, not the linear-storage default matrix-free CG guarantee. Direct solves validate the actual trust-ball boundary and original-model stationarity before returning a step. PCG controls are inactive under direct factorization; bandwidth is inactive when factorization is explicitly selected.",
"Trust-region defaults use 400 iterations, 100*n objective evaluations, and 1e-6 step/optimality/function tolerances. The dimension-dependent evaluation and PCG defaults can remain [] in constructor output; documented fixed defaults remain inspectable. Iterations count actual trial attempts, including rejected trials, and funcCount counts actual objective calls, including gradient-difference probes but not Hessian products. Trial H/Hinfo is committed only when its point is accepted; rejected owners retire without replacing the accepted model.",
"Trust-region OutputFcn states are init, iter after each actual trial attempt, and done. A rejected trial reports the retained accepted x, objective and gradient with updated actual counts/radius. Metadata contains iteration, funccount, fval, gradient, firstorderopt, stepsize, cgiterations and trustregionradius, without invented line-search quantities. Output.cgiterations counts actual PCG iterations; direct subproblems add zero PCG iterations. The sixth output is the last model actually constructed/used, or original Hinfo, not an automatically recomputed Hessian at final x. If early optimality, zero iterations or exhausted model budget prevents construction, it is [].",
"Trust-region flag 1 compares the actual evaluated gradient infinity norm to OptimalityTolerance; flag 2 uses a sufficiently small trust step; flag 3 uses sufficiently small accepted objective change; flag 5 uses sufficiently small positive model-predicted decrease. Flags 2, 3 and 5 do not assert first-order optimality. Flags 0 and -1 retain their limit/observer meanings. Display='iter' reports actual attempts, objective evaluations, CG iterations, trust radius and first-order optimality through the original console/diary route."]
},

BuiltinDocumentationSection {
 heading: "Quasi-Newton termination and iterative display",
 paragraphs: &["Exitflag 1 is returned only when the infinity norm of the current evaluated gradient is at most TolFun (OptimalityTolerance). Exitflag 2 means the step displacement is at most TolX times one plus the infinity norm of the previous iterate; this alone is not a first-order optimality certificate. Exitflag 0 means an iteration/evaluation limit, and -1 means an OutputFcn stop. Quasi-Newton does not stop merely because successive objective values are close. Different line searches and finite-difference gradients can still produce different stopping points or flags from another implementation; no Rosenbrock-specific flag is forced.",
"ObjectiveLimit defaults to -1e20. An actually evaluated objective/gradient state with fval <= ObjectiveLimit stops with -3, retaining that point and its actual gradient/call count. This threshold stop is not a proof of unboundedness. In RunMat extension mode only, numerical line-search or trust-region subproblem failure returns -4 with the last accepted x, fval and gradient, and an explicit nonconvergence message. Strict compatibility mode rejects this additional status with RunMat:compatibility:FminuncNumericalFailureExtension rather than attributing a MATLAB meaning to it.",
"Display='iter' writes one header, the initial evaluated point, one row per accepted iteration, and a terminal summary through the invocation's console capture, diary and forwarding route. Columns report Iteration, Func-count, f(x), Step-size (the actual line-search multiplier), and First-order optimality (the evaluated gradient infinity norm). Before the first step, Step-size is unavailable and is shown as a dash. Output.stepsize and OutputFcn optimValues.stepsize remain the actual iterate displacement, not the line multiplier. No objective evaluations or observer replays are added for display.",
"Display='final' is summary-only; 'notify' emits a summary only for nonpositive termination flags. 'off' and 'none' emit no solver presentation. The table is RunMat's textual presentation, not a claim of byte-identical MATLAB console layout."] }
,

BuiltinDocumentationSection {
 heading: "Execution and residency",
 paragraphs: &["Both optimization algorithms run on the host, without native gpuArray or tall-array solver execution. Authenticated Automatic placement can transparently materialize consumed numeric operands through their original invocation/provider. Explicit device intent uses the declared RunMat resident extension; unproven placement is refused before transfer. Opaque Hinfo is not treated as numeric data or recursively gathered.",
"`fminunc` is a host-side optimization sink. GPU-resident initial guesses are gathered before the solve, and objective callbacks may opt back into GPU work internally.",
"The canonical host-materialization boundary is required because solver decisions consume scalar objective values, gradients and active numerical models on the CPU.",
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
 },
BuiltinExample {id:"example-limited-memory",title:"Select a real limited-memory update",program:"fun = @(x) deal(sum((x-[1;2]).^2),2*(x-[1;2]));\nopts=optimoptions('fminunc','HessianApproximation',{'lbfgs',3},'SpecifyObjectiveGradient',true,'Display','off');\n[x,fval,flag,output,grad,H]=fminunc(fun,[0;0],opts);",display_output:None,compatibility:BuiltinExampleCompatibility::Matlab,harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,verification:BuiltinExampleVerification::Assertions{source:"assert(norm(x-[1;2])<1e-5); assert(fval<1e-8); assert(norm(grad)<1e-5); assert(strcmp(output.algorithm,'quasi-newton lbfgs')); assert(norm(H-2*eye(2))<1e-6);"}},
BuiltinExample {id:"example-trust-region",title:"Use a genuine supplied-Hessian trust-region model",program:"fun = @(x) deal(sum((x-[1;2]).^2),2*(x-[1;2]),2*eye(2));\nopts = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective','Display','off');\n[x,fval,flag,output,grad,H] = fminunc(fun,[0;0],opts);",display_output:None,compatibility:BuiltinExampleCompatibility::Matlab,harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,requirements:BuiltinExampleRequirements::NONE,verification:BuiltinExampleVerification::Assertions{source:"assert(norm(x-[1;2])<1e-5); assert(fval<1e-8); assert(norm(grad)<1e-5); assert(strcmp(output.algorithm,'trust-region')); assert(norm(H-2*eye(2))<1e-8);"}}
],

    example_exemption: None,

    faqs: &[BuiltinDocumentationFaq {
 question: "Does fminunc use the GPU?",
 answer: "The optimizer state and line search run on the host because the solver repeatedly invokes user callbacks. The callback itself can use GPU-aware operations." }
,

BuiltinDocumentationFaq {
 question: "What is in the output struct?",
 answer: "`output` contains `iterations`, `funcCount`, `algorithm`, `firstorderopt`, `stepsize`, and `message`; trust-region also reports actual `cgiterations`." }
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
