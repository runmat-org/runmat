use crate::*;
pub(super) const LATIN_SECTION: BuiltinDocumentationSection = BuiltinDocumentationSection { heading: "Latin-hypercube search", paragraphs: &[
            "SearchFcn='searchlhs' selects Latin-hypercube sampling before polling. The default sample count is 15*numberOfVariables and the default iteration limit is one search/poll pass. A cell {@searchlhs,iterlim,level}, or a cell with the method name first, supplies positive host scalar counts for the pass limit and sample count. The limit includes inner nonlinear subproblem passes and does not reset at multiplier updates. Native integer counts use the existing structural-option extension.",
            "Each coordinate uses an independent unbiased permutation and uniform jitter from the invocation stream. The candidate set is generated before callbacks; no random turn remains held while evaluating an objective or constraint. Finite bound pairs give the sampling interval, and fixed coordinates remain constant. With both bounds missing the interval is [-1,1]. For a one-sided bound, RunMat uses a width-two window extending from its finite endpoint, enlarged to an adjacent representable endpoint when needed; an extreme endpoint with no finite outward neighbor is constant. Original bounds and linear rows still decide which points can be evaluated.",
            "UseCompleteSearch=true evaluates the feasible candidates and retains the best measured merit. With false, the first improvement ends the search. Selected parallel execution uses complete search, and selected vectorized or worker waves keep the same remaining objective-point budget. Infeasible linear samples and already visited cached samples do not add objective calls. A failed or exhausted sample set cannot by itself contract the polling mesh or certify convergence. Search improvements use the selected poll strategy's mesh expansion.",
            "Latin search and mutable output callbacks retain the invocation stream before evaluating the objective. Exceptions and cancellation do not repeat evaluations. A zero point budget generates no samples and makes no search calls.",
        ]};
pub(super) const LATIN_EXAMPLE: BuiltinExample = BuiltinExample {
            id:"latin-bounded-search", title:"Search a bounded interval with Latin-hypercube points",
            program:"opts = optimoptions('patternsearch','Display','off','SearchFcn','searchlhs','UseCompleteSearch',true,'MaxFunctionEvaluations',16);\n[x,fval,flag,out] = patternsearch(@(x) -x,0,[],[],[],[],0,1,[],opts)",
            display_output:None, compatibility:BuiltinExampleCompatibility::Matlab,
            harness:BuiltinExampleHarness::Portable, fixture:BuiltinExampleFixture::None,
            requirements:BuiltinExampleRequirements::NONE,
            verification:BuiltinExampleVerification::Assertions { source:"assert(x>0.9 && x<=1); assert(fval==-x); assert(flag==0); assert(out.funccount==16);" },
        };
