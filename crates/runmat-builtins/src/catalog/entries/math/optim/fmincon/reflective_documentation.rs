use crate::*;

pub(super) const SECTION: BuiltinDocumentationSection = BuiltinDocumentationSection {
    heading: "Trust-region-reflective",
    paragraphs: &[
        "Algorithm='trust-region-reflective' requires SpecifyObjectiveGradient=true. It supports bounds alone, linear equalities alone, or an unconstrained problem. Linear inequalities, nonlinear constraints, combined bounds and linear equalities, and equal lower/upper bounds are not supported by this algorithm. Sparse matrices are supported. Bound starts are moved to a representable strict interior; equality starts are corrected using one retained sparse QR/COD factor. Every original equality row is checked, including redundant rows.",
        "The bound algorithm uses a Coleman–Li scaled quadratic model and compares bound-limited, reflected and scaled-Cauchy trajectories. The equality algorithm projects the trust-region model onto the original linear null space. Conjugate gradients uses operator products and a band preconditioner, not a hidden dense inverse. SubproblemAlgorithm='factorization' or PrecondBandWidth=Inf selects explicit direct factorization; its matrix workspace is quadratic in the number of variables. The original model residual and original constraint/KKT checks remain the acceptance authority.",
        "With HessianFcn=[], HessianMultiplyFcn=[] uses finite differences of supplied gradients. HessPattern is a symmetric real/logical dense or sparse structural pattern; graph coloring shares compatible probes. TypicalX and FiniteDifferenceStepSize do not apply to this algorithm. HessianFcn='objective' requests [f,g,H], with H a real symmetric matrix. Alternatively, HessianMultiplyFcn supplies W=hmul(Hinfo,Y), where [f,g,Hinfo]=fun(x). Hinfo is an original retained value, not converted into a numeric matrix. HessianMultiplyFcn requires HessianFcn=[] and the cg subproblem; direct factorization is not supported for this callback form. A selected objective Hessian leaves multiply/pattern controls dormant, and a selected multiply callback leaves the pattern dormant.",
        "Reflective defaults are MaxIterations=400, MaxFunctionEvaluations=100*numel(x0), OptimalityTolerance=1e-6, FunctionTolerance=1e-6, StepTolerance=1e-6, SubproblemAlgorithm='cg', TolPCG=0.1 and PrecondBandWidth=0. MaxPCGIter defaults to max(1,floor(numel(x0)/2)) for bounds and numel(x0) for equalities. PrecondBandWidth accepts a nonnegative integer or Inf; MaxPCGIter and TolPCG are inactive in direct mode. Legacy TolFun names the function-change criterion here, independently of modern OptimalityTolerance. Conflicting bindings to the same criterion are refused. Known concrete defaults are inspectable; dimension-dependent defaults are resolved when the problem size is known.",
        "Reflective output adds cgiterations and trustregionradius to the ordinary information fields. Flag 2 tests an accepted feasible relative step, and flag 3 tests accepted feasible relative function change; neither promises first-order optimality. The seventh output is the last-used Hessian model, or the original Hinfo for HessianMultiplyFcn, without building an unused successor or replaying callbacks. It is [] if no model was used. All gradient/Hessian probes and accepted or undefined evaluated trials obey the original hard point budget; service errors and callback exceptions are never numerical radius retries.",
    ],
};

pub(super) const BOUNDS: BuiltinExample = BuiltinExample {
    id: "reflective-bounded-gradient",
    title: "Use a supplied gradient with reflective bounds",
    program: "options=optimoptions('fmincon','Algorithm','trust-region-reflective',...\n'SpecifyObjectiveGradient',true,'Display','off');\n[x,fval,flag]=fmincon(@boundedquad,[0.5;0.5],[],[],[],[],[0;0],[1;2],[],options)\nfunction [f,g]=boundedquad(x)\nf=(x(1)-2)^2+(x(2)-1)^2;\ng=2*(x-[2;1]);\nend",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(flag>0); assert(norm(x-[1;1],Inf)<1e-4); assert(abs(fval-1)<1e-4); assert(all(x>=[0;0])); assert(all(x<=[1;2]));",
    },
};

pub(super) const EQUALITY: BuiltinExample = BuiltinExample {
    id: "reflective-sparse-equality-hessian",
    title: "Use sparse equalities and an objective Hessian",
    program: "options=optimoptions('fmincon','Algorithm','trust-region-reflective',...\n'SpecifyObjectiveGradient',true,'HessianFcn','objective',...\n'SubproblemAlgorithm','factorization','Display','off');\n[x,fval,flag]=fmincon(@equalityquad,[0;0],[],[],sparse([1 1]),2,[],[],[],options)\nfunction [f,g,H]=equalityquad(x)\nf=sum((x-[3;2]).^2);\ng=2*(x-[3;2]);\nH=sparse(2*eye(2));\nend",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(flag>0); assert(norm(x-[1.5;0.5],Inf)<1e-4); assert(abs(fval-4.5)<1e-4); assert(abs(sum(x)-2)<1e-6);",
    },
};
