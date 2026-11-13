use crate::*;
pub(super) const SECTION: BuiltinDocumentationSection = BuiltinDocumentationSection {
    heading:"Nelder-Mead search",
    paragraphs:&[
        "SearchFcn='searchneldermead' runs a Nelder-Mead simplex search before polling, for unconstrained problems only. Bounds and linear or nonlinear constraints are rejected before the first objective evaluation. Explicit infinite bounds do not impose finite restrictions. A cell {@searchneldermead,iterlim,optionsNM} sets the positive number of Pattern passes that run this search (default one) and a scalar fminsearch options structure from optimset.",
        "The simplex starts from the current measured point without evaluating it again. It runs to its own TolX/TolFun, MaxIter/MaxFunEvals, output-function or plot-function stopping condition, while sharing the remaining Pattern objective budget, original context, cancellation and time limit. Local counts describe only additional objective attempts. A partial limit retains an actual earlier improvement; a local Nelder-Mead stop does not certify Pattern convergence.",
        "optionsNM supports the same implemented fminsearch controls: Display, FunValCheck, MaxIter, MaxFunEvals, TolX, TolFun, OutputFcn, and PlotFcns. Its callbacks receive normal fminsearch iteration metadata and phases. A nested observer stop ends the auxiliary search, not the outer Pattern solve. FunValCheck='on' rejects undefined auxiliary trials; otherwise undefined or skipped cached trials cannot establish simplex convergence. UseCompleteSearch does not apply.",
        "Selected vectorized and worker modes retain their actual callback route, including one-point measurements. The simplex arithmetic, order and observers do not replay an objective for reporting or verification. Original callback exceptions and cancellation propagate rather than becoming unsuccessful simplex steps. Nelder-Mead remains a local derivative-free heuristic, not a global or first-order optimality certificate.",
    ],
};
pub(super) const EXAMPLE:BuiltinExample=BuiltinExample {
    id:"known-seed-nelder-search",title:"Use a one-evaluation auxiliary simplex budget",
    program:"local=optimset('Display','off','MaxFunEvals',1);\nopts=optimoptions('patternsearch','Display','off','SearchFcn',{@searchneldermead,1,local},'MaxIterations',1);\n[x,fval,flag,out]=patternsearch(@(x) (x-1)^2,0,[],[],[],[],[],[],[],opts)",
    display_output:None,compatibility:BuiltinExampleCompatibility::Matlab,
    harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,
    requirements:BuiltinExampleRequirements::NONE,
    verification:BuiltinExampleVerification::Assertions {
        source:"assert(x==0.00025); assert(abs(fval-(x-1)^2)<1e-12); assert(out.funccount==2);",
    },
};
