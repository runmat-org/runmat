use super::*;
pub(super) const SECTION: BuiltinDocumentationSection = BuiltinDocumentationSection {
    heading: "Nonlinear least-squares models",
    paragraphs: &[
        "A recognized minimized sum of positive weighted squared nonlinear residuals selects lsqnonlin when the model has only variable bounds. Supply a starting-point structure containing the original variable names and shapes. Residual rows follow the original term order and column-major array order; solver probes evaluate the retained expression graph.",
        "The returned fval is the original scalar objective, including additive constants, not the solver's residual norm. The solution retains all original variable fields and shapes. output.Solver identifies lsqnonlin; its iteration and function counts include the residual and finite-difference probes that were evaluated. Final derivative probes do not replace the accepted solution point.",
        "With MaxFunctionEvaluations=0, RunMat returns the bound-projected supplied variable fields without evaluating the original graph. fval is NaN, the limit flag and function count are zero, and multipliers are empty. The information identifies not-started and not-evaluated rather than an accepted numerical point.",
        "Explicit lsqnonlin selection rejects comparison constraints before residual work. A constrained residual model can still use the scalar fmincon route, with those constraints preserved. Options must belong to the selected solver; an incompatible options profile is not reinterpreted. Least-squares solvers do not maximize, and integer or semi domains are not relaxed.",
        "The residual kernel currently uses damped Levenberg-Marquardt steps. Both accepted raw algorithm selectors currently reach this kernel; a separate trust-region-reflective residual mechanism is not yet available. Use Algorithm='levenberg-marquardt' to identify the implemented mechanism explicitly.",
        "Local convergence flags describe the numerical stopping condition, not a global minimum certificate. The original objective and residuals must be finite and real at evaluated points. Operational errors and cancellation propagate without replacement residuals or a retry of completed work. Bound multipliers use the full original squared-objective units; unavailable final derivatives leave the multiplier fields empty rather than inventing zero derivatives.",
    ],
};
pub(super) const EXAMPLE: BuiltinExample = BuiltinExample {
    id: "nonlinear-residual-source-objective",
    title: "Solve nonlinear residuals and retain the objective constant",
    program: concat!(
        "x=optimvar('rate');\n",
        "p=optimproblem('Objective',(sin(x)-0.5)^2+7); x0.rate=0.3;\n",
        "o=optimoptions('lsqnonlin','Algorithm','levenberg-marquardt','Display','off');\n",
        "[s,f,e,out]=solve(p,x0,'Options',o);"
    ),
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(abs(s.rate-asin(0.5))<1e-5); assert(abs(f-7)<1e-9); assert(double(e)>0); assert(strcmp(out.Solver,'lsqnonlin')); assert(out.funcCount>1); assert(abs(evaluate(p.Objective,s)-f)<1e-12);",
    },
};
