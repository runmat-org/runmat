use crate::*;
pub(super) const DOCUMENTATION:BuiltinDocumentation=BuiltinDocumentation{
    authority:BuiltinDocumentationAuthority::Catalog,
    title:Some("fseminf"),slug:Some("fseminf"),
    summary:"Find a local minimum subject to finite and semi-infinite constraints.",
    description:"fseminf uses adaptive one- or two-dimensional constraint interpolation and genuine working-set sequential quadratic programming. Mesh generations and their constraint/multiplier identities are refreshed at accepted points before first-order or termination publication.",
    keywords:&["fseminf","semi-infinite","adaptive mesh","SQP","constrained optimization"],
    related:&["fmincon","optimoptions","optimset"],
    sections:&[
        BuiltinDocumentationSection{heading:"Calling forms and data",paragraphs:&[
            "Use fseminf(fun,x0,ntheta,seminfcon), optionally followed by complete A/b, Aeq/beq and lb/ub pairs, then options. Empty pairs omit finite constraints and empty bounds omit their side. The scalar problem structure requires objective,x0,ntheta,seminfcon,solver='fseminf',options, with optional Aineq,bineq,Aeq,beq,lb,ub.",
            "Numeric problem inputs and numeric callback results are real host double data. x0 is finite and nonempty, ntheta is a positive integer-valued double scalar, and every callback and returned x preserves the original x0 shape. Linear matrices multiply column-major x(:). Sparse, single, native-integer/logical and explicit device problem data are not silently converted into documented double data.",
            "Bounds may contain correctly signed infinities but not NaN. Shorter nonempty bound arrays constrain leading components and warn; remaining coordinates are unbounded. Starting points are clipped to consistent bounds. Inconsistent bounds return original x0, fval=[] and a local infeasibility exit, without invoking the objective or constraints.",
        ]},
        BuiltinDocumentationSection{heading:"Semi-infinite callback",paragraphs:&[
            "fun returns a real scalar f. SpecifyObjectiveGradient=true requests [f,g], with one gradient element per variable. seminfcon has signature [c,ceq,K1,...,Kntheta,S]=seminfcon(x,S). c<=0 and ceq=0 are finite vector constraints with fixed cardinalities. Their gradients and every semi-infinite gradient use the selected finite-difference owner; no analytic infinite-family Jacobian contract is invented.",
            "The initial S is an ntheta-by-2 NaN matrix. Each returned Ki is a nonempty finite double vector or matrix sampled on a fixed interval or rectangle, and returned S reports the actual positive sampling intervals. A vector uses S(i,1) only; its dormant second interval is not read. Matrix columns use S(i,1), rows use S(i,2). The origin and physical domain must remain fixed across x and mesh recommendations.",
            "S supplied to subsequent accepted-point refreshes recommends refinement or coarsening, not a mandatory data shape. A callback may ignore the recommendation; the solver remaximizes the actual returned interpolant and never pretends the requested mesh was produced. Trial/probe callbacks receive accepted actual intervals. If they return a different sampling of the same fixed physical domain, interpolation evaluates the same frozen physical peak identities on that actual mesh; the row generation is not retargeted. Accepted-point refreshes remaximize the new grid and remap row identities before any KKT publication.",
            "The functions must be continuous in x and their one or two independent family variables. The numerical discretization is not a proof of feasibility between arbitrary unseen features in the original callback function. Provide initial sampling appropriate to those functions and verify a result on independently meaningful domain data.",
        ]},
        BuiltinDocumentationSection{heading:"Numerical algorithm and ownership",paragraphs:&[
            "Each sampled cell uses adjacent quadratic or cubic one-dimensional/tensor interpolation. Bernstein coefficient bounds and cooperative subdivision locate continuous interpolant maxima, including off-grid peaks; sampled-node maxima are not substituted for the semi-infinite problem. Quadratic/cubic and held-out interpolation defects drive refinement recommendations; safely inactive dense families may coarsen. An exactly resolved off-grid peak is not treated as a sampling error.",
            "One genuine constrained SQP major step uses the current interpolated peak constraints, original finite rows and bounds. A primal working-set quadratic mechanism adds blocking inequalities and removes rows with negative duals. Independent original-model KKT residuals, nonlinear feasibility, dual signs and complementarity govern first-order publication. Incompatible linearizations use the existing single-elastic-variable maximum-violation restoration rather than an unconstrained fixed penalty.",
            "Before any initial or accepted-point KKT check, the accepted raw samples are remaximized and any recommended same-x mesh callback is charged and evaluated. The objective value and analytic gradient are reused only at that exact x. New constraint Jacobians use typed constraints-only probes, which cannot supply an objective value. Every actual objective and constraint invocation is counted separately; these probes do not replay fun.",
            "Opaque mesh generations stamp both accepted points and linearizations. Finite rows retain identity; only exactly matched physical peak identities receive warm duals. A warm remap is never an optimality certificate. Old-generation models are invalidated before asynchronous refresh, and no historical mesh archive or per-probe invocation-context copy is retained. Ordinary finite constrained solvers keep their existing domains unchanged.",
            "Changing peak rows do not receive an invalid cross-generation Lagrangian BFGS secant. The existing positive-semidefinite step model is retained across these refreshes. This can be slower than a tuned implementation and is not a claim of MATLAB-identical iteration history or performance.",
        ]},
        BuiltinDocumentationSection{heading:"Supported options and defaults",paragraphs:&[
            "ConstraintTolerance/TolCon defaults to 1e-6; OptimalityTolerance defaults to 1e-6; FunctionTolerance defaults to 1e-4; StepTolerance/TolX defaults to 1e-4. These tolerances are nonnegative. Legacy TolFun supplies both function and optimality roles; a conflicting explicit modern value is rejected. Function stopping measures the actual objective directional derivative at a feasible accepted model.",
            "MaxIterations/MaxIter defaults to 400 and MaxFunctionEvaluations/MaxFunEvals to 100*numel(x0); both permit zero. A complete derivative sweep and trial respect the hard evaluated-point budget. Same-x mesh refreshes can reuse fun but are real charged constraint evaluations. output.funcCount records evaluated points, with separate actual objectiveCallCount and constraintCallCount.",
            "MaxSQPIter is a positive inner working-set iteration limit, defaulting to 10*max(numberOfVariables,numberOfInequalities+numberOfBounds). TolConSQP is a positive inner constraint tolerance, default 1e-6. They control the actual quadratic mechanism rather than merely being accepted metadata.",
            "FiniteDifferenceType/FinDiffType is forward by default or central. FiniteDifferenceStepSize/FinDiffRelStep is a finite positive floating scalar or vector, with a scalar factor expanded to all variables. TypicalX is a finite positive floating vector with one element per variable. Empty defaults are sqrt(eps) for forward factors, eps^(1/3) for central factors and unit TypicalX.",
            "Forward steps use sign(x), with sign(0)=1, times the relative factor and max(abs(x),TypicalX). Central steps omit that sign. DiffMinChange defaults to 0 and DiffMaxChange to Inf and clamp the absolute nominal magnitude. Actual representable bound-feasible offsets own the denominators; unusable offsets or overflowing central spans refuse before callbacks. Fixed coordinates use feasible-direction differences, not invented full objective derivatives.",
            "RelLineSrchBnd is [] by default or a finite nonnegative scalar limiting each displacement by that factor times max(abs(x),TypicalX). RelLineSrchBndDuration is a positive count, default 1, of major iterations for which the bound applies. Bounds are checked before trial callbacks.",
            "Diagnostics and FunValCheck accept on/off, default off. Diagnostics emits actual model/call information. FunValCheck=on errors for complex, Inf or NaN objective/constraint results. With it off, only actual undefined numerical trials may trigger smaller trials; invalid carriers, callback exceptions and cancellation remain typed errors.",
            "Display accepts off, none, iter, iter-detailed, notify, notify-detailed, final and final-detailed; default final. Detailed display includes real tolerance information. OutputFcn accepts a handle or cell of handles; PlotFcn (legacy PlotFcns) accepts handles, names or cells. init/iter/done callbacks use the original execution context and may request stop without replaying objective data.",
            "All seven predefined plots are real shared callbacks: optimplotx, optimplotfunccount, optimplotfval, optimplotfvalconstr, optimplotconstrviolation, optimplotstepsize and optimplotfirstorderopt. The best-feasible plot uses feasibility tolerance 1e-6 and distinct feasible/infeasible colors. Plotting requires plot-core and UI authority; it never silently succeeds without a graphics implementation.",
            "Unsupported nonempty options are rejected. There is no accepted Algorithm, UseParallel or SpecifyConstraintGradient placeholder for this solver. Native-integer scalar options and explicit device option readback use their named RunMat extensions; these do not broaden the strict double problem/callback data boundary.",
        ]},
        BuiltinDocumentationSection{heading:"Outputs and execution",paragraphs:&[
            "Up to five outputs are [x,fval,exitflag,output,lambda]. They use the last accepted original state without callback replay. fval is [] before any accepted objective value. Lambda is [] until a real independent estimate is available; otherwise lower,upper,ineqlin,eqlin,ineqnonlin,eqnonlin preserve original finite row/variable identities and exclude generated interpolated-peak rows.",
            "Exitflag 1 requires current-mesh independent nonlinear KKT success; 4 means a feasible small direction or allowed step; 5 means a feasible small objective directional derivative; 0 is a hard iteration/evaluated-point limit; -1 is observer stop; -2 is local restoration failure, not a global nonlinear infeasibility proof. Unreliable numerical geometry raises an error instead of inventing an undocumented success or zero-gradient fallback.",
            "output contains iterations,funcCount,lssteplength,stepsize,algorithm,constrviolation,firstorderopt,message plus separate actual objectiveCallCount/constraintCallCount. Requested output 4 also captures wallclocktime through the original admitted Clock port; fewer outputs do not read the clock. No ambient clock or runtime context is manufactured.",
            "This is a cooperative host numerical solver on native and WebAssembly targets. Callback effects, cancellation, service authority and output roots remain those of the original invocation. Finite host-memory execution refuses before solver work because dynamic interpolation/factorization/graphics workspaces have no complete finite bound. No GPU optimizer, distributed execution or global minimum certificate is claimed.",
        ]},
    ],
    examples:&[
        BuiltinExample{
            id:"off-grid-parabolic-family",title:"Resolve an infinite-family peak between samples",
            program:"fun=@(x)(x-1)^2;\noptions=optimoptions('fseminf','Display','off');\n[x,fval,flag]=fseminf(fun,0.2,1,@family,[],[],[],[],0,2,options)\nfunction [c,ceq,K,S]=family(x,S)\nif isnan(S(1,1)); S=[0.25 0]; end\nt=0:S(1,1):1; c=[]; ceq=[]; K=x-0.5-(t-0.5).^2;\nend",
            display_output:None,compatibility:BuiltinExampleCompatibility::Matlab,
            harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,
            requirements:BuiltinExampleRequirements::NONE,
            verification:BuiltinExampleVerification::Assertions{
                source:"assert(flag>0); assert(abs(x-0.5)<1e-4); assert(abs(fval-0.25)<1e-4);"},
        },
        BuiltinExample{
            id:"inconsistent-bounds-no-callback",title:"Return the original point for inconsistent bounds",
            program:"[x,fval,flag]=fseminf(@(x)error('objective must not run'),3,1,@never,[],[],[],[],4,2)\nfunction [c,ceq,K,S]=never(x,S)\nerror('constraint must not run'); c=[]; ceq=[]; K=[];\nend",
            display_output:None,compatibility:BuiltinExampleCompatibility::Matlab,
            harness:BuiltinExampleHarness::Portable,fixture:BuiltinExampleFixture::None,
            requirements:BuiltinExampleRequirements::NONE,
            verification:BuiltinExampleVerification::Assertions{
                source:"assert(x==3); assert(isempty(fval)); assert(flag==-2);"},
        },
    ],
    example_exemption:None,faqs:&[],media:&[],
    links:&[
        BuiltinDocumentationLink{label:"MATLAB fseminf",target:BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/optim/ug/fseminf.html")},
        BuiltinDocumentationLink{label:"Semi-infinite constrained algorithms",target:BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/optim/ug/constrained-nonlinear-optimization-algorithms.html")},
    ],
    evidence:BuiltinDocumentationEvidence{
        implementation:&[BuiltinDocumentationLink{label:"Dynamic semi-infinite numerical owner",
            target:BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/fmincon/semi_infinite/mod.rs")}],
        verification:&[BuiltinEvidenceReference{kind:BuiltinEvidenceKind::UnitTest,
            label:"Continuous interpolation extrema, mesh generations and original-call accounting",
            location:"builtins::math::optim::fmincon::semi_infinite"}],
        notes:&[],
    },
    introduced:None,status:Some(BuiltinDocumentationStatus::Partial),
};
