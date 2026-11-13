use crate::*;
pub(super) const SECTION: BuiltinDocumentationSection = BuiltinDocumentationSection {
    heading:"Radial-basis surrogate search",
    paragraphs:&[
        "SearchFcn='rbfsurrogate' forms a cubic radial-basis interpolant with an affine polynomial tail from recent finite evaluations. The search starts after at least n+1 distinct points, where n is numel(x0). After every 10*n actual objective attempts, the model and retained samples are erased; sampling restarts with subsequent points. Undefined trials count toward that reset but are not interpolation sites. Repeated coordinates keep their most recent finite measurement.",
        "Explicit finite lower and upper bounds are retained. A missing endpoint is estimated from the recent coordinate minimum or maximum. Original linear constraints and bounds still restrict every evaluated proposal. Equality-constrained models keep full physical distances and use independent coordinates only for the affine tail. Fixed coordinates do not create artificial independent tail columns.",
        "For nonlinear constraints, raw objective and constraint values are retained, and interpolation scores use the current augmented-Lagrangian multipliers and penalty. A multiplier update does not reuse stale merit scores. Coordinate arrays and constraint payloads share their immutable callback backing; no callback is repeated to fill a model or verify a proposed point.",
        "RunMat minimizes the interpolant locally using feasible generating directions and box-clipped steps, with at most 8*(n+1) direction attempts per search. This is a proposal-work policy rather than a stopping certificate for the original objective. Singular, undefined or unpoised models skip the auxiliary phase. A new proposal is evaluated once through the selected scalar, vectorized or worker route, and accepted only on an actual merit improvement. UseCompleteSearch does not apply.",
    ],
};
pub(super) const EXAMPLE:BuiltinExample=BuiltinExample {
    id:"rbf-surrogate-search",title:"Use recent samples for a radial-basis proposal",
    program:"opts=optimoptions('patternsearch','Display','off','SearchFcn','rbfsurrogate','ScaleMesh',false,'MaxFunctionEvaluations',100);\n[x,fval,flag,out]=patternsearch(@(x) (x-1.2)^2,0,[],[],[],[],0,3,[],opts)",
    display_output:None,compatibility:BuiltinExampleCompatibility::RunMat,
    harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,
    requirements:BuiltinExampleRequirements::NONE,
    verification:BuiltinExampleVerification::Assertions {
        source:"assert(x>=0 && x<=3); assert(abs(x-1.2)<0.05); assert(abs(fval-(x-1.2)^2)<1e-12); assert(out.funccount<=100);",
    },
};
