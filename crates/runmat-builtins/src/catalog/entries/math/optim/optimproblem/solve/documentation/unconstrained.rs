//! Unconstrained original-model consumer.
use super::*;
pub(super) const UNCONSTRAINED: BuiltinDocumentationSection = BuiltinDocumentationSection {
    heading: "Unconstrained nonlinear models",
    paragraphs: &[
        "General scalar continuous nonlinear models without bounds or comparison constraints select fminunc. Explicit Solver='fminunc' requires this same geometry and does not discard bounds, comparisons or integer domains. A starting-point structure supplies each original variable in its original shape; returned solution fields preserve those shapes.",
        "This consumer uses the fminunc quasi-Newton controller, including BFGS, limited-memory BFGS, and the applicable legacy update controls. The original expression DAG supplies scalar objective values; finite differences use the shared forward/central refinement policy, selected step sizes and TypicalX. Maximization changes only the solver objective sign, and fval remains the original objective.",
        "Options uses the fminunc profile and its actual function/iteration limits, tolerances, Display and OutputFcn controls. Supplied gradients and modeled parallel derivatives are unavailable and are rejected before starting-point conversion. Requesting trust-region without supplied gradients selects quasi-Newton with the same warning as fminunc; inactive trust-region controls are not inspected. Model evaluation errors and nonfinite or nonreal objective values follow the raw fminunc error policy, rather than being replaced by artificial trial values.",
        "output reports iterations, funcCount, firstorderopt, stepsize, algorithm, message, Solver and objectivederivative='finite-differences'. This information does not read Clock. lambda is []. ObjectiveLimit stops at the evaluated threshold-crossing iterate; its public Unbounded (-3) exitflag is a threshold-stop projection, not a mathematical unboundedness certificate. The message identifies the threshold condition.",
    ],
};
pub(super) const EXAMPLE: BuiltinExample = BuiltinExample {
    id: "unconstrained-original-model",
    title: "Minimize an unconstrained nonlinear model from its original starting point",
    program: concat!(
        "x=optimvar('rate',2,1);\n",
        "p=optimproblem('Objective',(x(1)-1)^4+(x(2)-2)^2);\n",
        "x0.rate=[0;0]; o=optimoptions('fminunc','Display','off');\n",
        "[s,f,e,out,lambda]=solve(p,x0,'Options',o);"
    ),
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isequal(size(s.rate),[2 1])); assert(abs(s.rate(1)-1)<0.05); assert(abs(s.rate(2)-2)<1e-4); assert(f<1e-5); assert(double(e)>0); assert(strcmp(out.Solver,'fminunc')); assert(isempty(lambda)); assert(abs(evaluate(p.Objective,s)-f)<1e-12);",
    },
};
