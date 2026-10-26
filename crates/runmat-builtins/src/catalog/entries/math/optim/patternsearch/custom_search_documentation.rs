use crate::*;
pub(super) const SECTION: BuiltinDocumentationSection = BuiltinDocumentationSection {
    heading: "Custom search callbacks",
    paragraphs: &[
        "SearchFcn=@search calls [success,xBest,fBest,funccount]=search(fun,x,A,b,Aeq,beq,lb,ub,optimValues,options). The original handle and its captured values select the callback. A same-named user function takes the custom route rather than a predefined search. Custom cell parameters are not supported; bind extra parameters in the supplied handle.",
        "fun evaluates the current search subproblem through the original objective and nonlinear-constraint callbacks. For a nonlinear problem it returns the augmented-Lagrangian merit, not the raw objective. Original bounds and linear rows remain enforced. The current x reuses its measured generation without another callback. Undefined or cached trials return NaN; they cannot become newly measured improvements.",
        "optimValues contains x, fval, iteration, funccount, scale, problemtype, meshsize, and method. problemtype describes the current unconstrained, bound-constrained, or linearly constrained subproblem. success must be a logical scalar. A successful result must identify the best point and merit actually measured during this search, or the unchanged current point. funccount must equal the actual additional objective attempts. Returned claims never trigger validation evaluations or change the solver counter.",
        "UseVectorized follows the original x0 orientation: row/scalar x0 uses one point per matrix row and a merit column; column x0 uses one point per matrix column and a merit row. The same point axis applies to a single point and complete batches. Selected worker execution evaluates original objective and constraint targets through the existing worker path. Both modes share the remaining objective-point budget. UseCompleteSearch is inapplicable to custom callbacks: the callback controls its own search. A time or point limit stops the nested evaluator without authorizing more work.",
        "The supplied fun handle is valid only while this search callback runs. It can be assigned or captured locally, but invoking an escaped handle after the callback completes raises ScopedCallable:Expired. It cannot be sent to a worker, foreign adapter, or persistent codec. Its class is function_handle; functions reports scopedcallable, and func2str returns descriptive text rather than a reconstructible function name. Original exceptions can be caught by the search callback; uncaught exceptions and cancellation stop the solver without replay.",
    ],
};
pub(super) const EXAMPLE: BuiltinExample = BuiltinExample {
    id: "custom-search",
    title: "Measure a custom search point through the supplied function",
    program: "opts=optimoptions('patternsearch','Display','off','SearchFcn',@searchpoint,'MaxIterations',1);\n[x,fval,flag,out]=patternsearch(@(x) (x-0.25)^2,0,[],[],[],[],[],[],[],opts);\nfunction [success,xBest,fBest,funccount]=searchpoint(fun,x,A,b,Aeq,beq,lb,ub,optimValues,options)\nxBest=0.25; fBest=fun(xBest); funccount=1; success=fBest<optimValues.fval;\nend",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(x==0.25); assert(fval==0); assert(out.funccount==2);",
    },
};
