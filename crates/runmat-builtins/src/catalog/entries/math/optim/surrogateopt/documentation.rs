use crate::*;
mod examples;
pub(super) const SOLVER: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("surrogateopt"), slug: Some("surrogateopt"),
    summary: "Minimize an expensive objective with cubic surrogate models and bounded global sampling.",
    description: "Search a finite bounded domain with optional integer, linear and nonlinear inequality constraints, retaining actual evaluations and resumable checkpoints.",
    keywords: &["surrogate", "global optimization", "radial basis", "integer", "checkpoint"],
    related: &["surrogateoptplot", "optimoptions", "fmincon", "intlinprog", "rng"],
    sections: &[
        BuiltinDocumentationSection { heading: "Calling forms and inputs", paragraphs: &[
            "Use surrogateopt(objconstr,lb,ub), add intcon as a fourth input, or use the full form \
            surrogateopt(objconstr,lb,ub,intcon,A,b,Aeq,beq). Append an options structure to any of these forms. Missing integer or linear \
            constraints use []. Bounds are required, nonempty finite real double vectors with the same number of elements. Their elements are \
            read in column-major order. Equal bounds fix a coordinate.",
            "intcon contains one-based integer-valued double indices into the original coordinates. Duplicate indices are normalized. The \
            original inequalities are A*x(:)<=b and equalities are Aeq*x(:)=beq; dense and sparse real double matrices are accepted. A row's \
            right-hand side must match its matrix. Bounds, linear data, decision points and callback measurements do not have logical, single \
            or native-integer overloads.",
            "surrogateopt(problem) takes a scalar structure with objective,lb,ub,solver='surrogateopt',options. Optional fields are \
            intcon,Aineq,bineq,Aeq,beq,rngstate. Unknown fields are rejected. rngstate is a valid rng snapshot applied to the current original \
            random stream before numerical work. A checkpoint can instead be continued with surrogateopt(checkpointFile) or \
            surrogateopt(checkpointFile,opts).",
        ] },
        BuiltinDocumentationSection { heading: "Objective and constraint results", paragraphs: &[
            "objconstr is an actual function handle or scalar function name selected at entry. A scalar call receives a 1-by-nvars row. It \
            returns either a real double scalar Fval or one scalar structure with Fval and Ineq. Ineq is a real double vector interpreted as \
            Ineq<=0. Fval=[] selects feasibility-only search; there are no nonlinear equality constraints. The objective-presence and \
            inequality-count schema must remain fixed across actual calls.",
            "Fval and Ineq may contain NaN to mark an unusable measured row. Such a row is retained in trials but does not train a surrogate \
            or establish an incumbent. Infinite measurements, complex values, invalid classes/shapes, callback errors and cancellation are \
            errors. Sparse real double measurements are decoded without a dense mirror. Feasibility-only search continues until a stopping \
            condition; it does not automatically stop at its first feasible point.",
            "With UseVectorized=true and requested UseParallel='off', the callback receives M-by-nvars rows and returns an M-by-1 Fval column, \
            or a structure with that column and M-by-q Ineq. Fval=[] still denotes an absent objective. M is at most BatchUpdateInterval and \
            can be one; the final batch can be shorter. Each row consumes one function evaluation, not one batch. Missing initial measurement \
            lanes use the same actual scalar, vectorized or worker route.",
        ] },
        BuiltinDocumentationSection { heading: "Surrogate search", paragraphs: &[
            "Construction obtains distinct feasible sites and fits a cubic radial-basis interpolant with an affine tail. Distances use full \
            normalized physical coordinates. Fixed coordinates and exact linear equalities reduce the independent affine tail and sampling \
            span without changing the radial metric. Factorizations use the shared LU/QR owners; no inverse or dense normal-equation model is \
            formed.",
            "Adaptive search alternates random, crossover, random-orthogonal and coordinate proposals. Merit balances predicted improvement \
            against distance from evaluated sites, with a changing sampling scale. Small spacing or stalled construction can reset the \
            surrogate phase. The measured history, lifetime best and original evaluation allowance survive resets. Work accepted before a \
            reset keeps its original phase and can improve the returned best without being relabeled as new-phase training data.",
            "Bounds, original sparse linear rows and integer membership are checked before an expensive call. Mixed-integer repair uses the \
            existing feasibility owner. Modeled refinement uses SQP subproblems with the same Clock and CPU cooperation. RunMat's internal \
            branch selection limits a refinement search to 30 or 60 actual branch steps; a finite search that finds no usable proposal is not \
            an infeasibility certificate. Numerical models propose points but never supply public objective values.",
        ] },
        BuiltinDocumentationSection { heading: "Options and stopping", paragraphs: &[
            "Display is off/none/iter/final, default final. MaxFunctionEvaluations is a nonnegative exact count, default max(200,50*nvars). \
            MaxTime defaults to Inf and accepts nonnegative seconds or a genuine scalar duration. ObjectiveLimit defaults to -Inf; a feasible \
            actual objective strictly below it stops search. ConstraintTolerance defaults to 1e-3 and applies to nonlinear inequalities, not \
            the original linear/integer feasibility owner.",
            "BatchUpdateInterval is positive, default 1. It controls the measured model/observer update interval and the vectorized row \
            target. MinSurrogatePoints defaults to max(20,2*nvars), must be at least nvars+1, and is effectively at least BatchUpdateInterval. \
            MinSampleDistance is nonnegative, default 1e-6; zero permits exact-site adaptive revisits rather than imposing a hidden positive \
            spacing rule. InitialPoints, CheckpointFile, OutputFcn, PlotFcn, UseParallel and UseVectorized are described below. Other solver \
            options are rejected.",
            "InitialPoints is an M-by-nvars real double matrix or a scalar structure with X and optional Fval and Ineq. Complete supplied \
            measurements consume no objective allowance. Partial fields are retained only at the exact unchanged supplied point. If repair \
            changes a point's bits, all supplied measurement lanes for that point are discarded. Missing schema is learned from a real new \
            evaluation, not a callback replay. Known rows can determine the returned solution even with MaxFunctionEvaluations=0.",
            "Flags are 10 for a unique feasible point, 3 when too few usable new sites remain, 1 for ObjectiveLimit, 0 for the evaluation/time \
            boundary, -1 for a logical observer stop, and -2 when no feasible point is found. An observer stop retains -1 even before \
            feasibility. Inconsistent bound or integer-bound intervals return x=[] and fval=[] without objective, RNG, Clock, UI, pool or \
            checkpoint-file effects. Time checks are cooperative boundaries, not hard preemption of an in-flight callback or numerical kernel.",
        ] },
        BuiltinDocumentationSection { heading: "Parallel work and observers", paragraphs: &[
            "UseParallel is off/auto/on; false is off and true is auto. Requested parallel mode takes precedence over UseVectorized, including \
            when auto has no pool and uses scalar serial execution. Inactive vectorized operands are not gathered or inspected for provider \
            facts. On requires an actual selected worker route. Auto falls back only before submission for known ineligibility or missing \
            authority. Errors or partial acceptance are not replayed serially.",
            "Parallel execution uses one original-source CallbackBatch and the actual pool width. A FIFO retains a small surplus of \
            unsubmitted proposals; only pool-width accepted calls run concurrently. Completions update the measured history in actual \
            completion order. Cancellation and failures propagate as errors while accepted work retains its source, receipt and physical \
            retirement owners.",
            "OutputFcn and PlotFcn are handles/names or cells of them, called as stop=f(x,optimValues,state). state is init/iter/done and \
            every stop result must be a logical scalar. x is the best actual point, or []. optimValues contains \
            funccount,elapsedtime,fval,constrviolation,ineq,incumbentX/Fval/Ineq/Constrviolation,currentX/Fval/Ineq/Constrviolation, the \
            corresponding sample flags, feasibleCount,surrogateReset,surrogateResetCount,iteration and constraintTolerance. Calls occur at \
            measured update boundaries. A done return is validated without changing the completed termination reason.",
            "PlotFcn defaults to optimplotfvalconstr; [] disables plotting. Selected predefined charts include \
            optimplotfval,optimplotfvalconstr,optimplotx,optimplotfunccount and surrogateoptplot. They use actual receiver-backed graphics \
            scenes. Missing plot/UI service is an error when a chart is selected, not a hidden no-op. Custom observers retain their originally \
            selected definitions and rooted return payloads through stop decoding.",
        ] },
        BuiltinDocumentationSection { heading: "Checkpoint save and resume", paragraphs: &[
            "CheckpointFile defaults to empty. A nonempty path selects the original filesystem provider; a missing extension adds .mat. Saving \
            uses an exclusive staged replacement and one publish decision. It occurs after an iteration when at least ten solver seconds have \
            elapsed since the previous saved cut, and after final callbacks. A save cut drains already accepted work without submitting queued \
            surplus. Partial measured batches and all accepted results are included; the live continuation retains its unsubmitted proposals.",
            "A checkpoint is a RunMat Level-5 MAT file with one versioned uint8 record, not a MATLAB checkpoint file. It contains one measured \
            history, the original sparse domain, prepared initial template, actual callback code/captures, total point evaluations, saved \
            solver time, RNG state and small observer-frame metadata. Resume checks the execution environment and admits the saved original \
            code/source/static/class graph under the current original context. Old RuntimeContexts, worker grants, cancellation tokens, clocks \
            and service handles are not restored.",
            "Resume keeps total limits: a saved count of 30 with MaxFunctionEvaluations=100 leaves at most 70 new point evaluations. Saved \
            measured objective values are not reevaluated. Initial rows already saved are matched once, while unsubmitted repeated occurrences \
            retain their own calls. Internal models and sampling trajectories are rebuilt, so the later trajectory can differ from \
            uninterrupted execution. Current callbacks replay saved observer frames without objective calls, then receive an iteration marked \
            checkpointResume; checkpointReplay identifies historical frames. surrogateoptplot displays actual resume reference lines.",
            "Only \
            BatchUpdateInterval,CheckpointFile,Display,MaxFunctionEvaluations,MaxTime,MinSurrogatePoints,ObjectiveLimit,OutputFcn,PlotFcn,UseParallel \
            and UseVectorized can change on resume. A full options structure may repeat unchanged frozen controls and InitialPoints. A dormant \
            vectorized value that is not portable must be replaced explicitly before activating vectorization. CheckpointFile='' disables \
            subsequent saves. Failure before publication leaves the previous checkpoint. An interrupted publish reports its actual \
            disposition, including whether publication occurred. Cancellation can prevent a final save. Work after the last successful \
            saved cut is not recoverable as a committed checkpoint.",
            "Checkpoint captures support sessionless scalar/array, selected object and lexical-cell payloads with their admitted code. \
            Declared foreign interop and external/resident/distributed service references require a fresh portable service projection and are \
            rejected by this checkpoint path. A serialized display name, digest or old session ID never substitutes for that projection. \
            Version/environment mismatches are errors; records are not loaded into the workspace and do not run object load hooks.",
        ] },
        BuiltinDocumentationSection { heading: "Outputs and execution services", paragraphs: &[
            "Up to five outputs are x,fval,exitflag,output,trials. x is the best measured feasible row, or the row of least maximum nonlinear \
            infeasibility when none is feasible. fval is its actual scalar measurement, or [] for an absent objective or absent point. output \
            contains funccount,elapsedtime,message,constrviolation,ineq,rngstate. rngstate is the initial state of the original stream; \
            resumed output retains the original saved initial state. trials.X has actual point rows, trials.Fval their column or [], and \
            trials.Ineq their measured inequality rows. Publication does not call the objective.",
            "Numerical work uses the original RuntimeContext, CPU budget, cancellation, Clock and random stream. Host decision data and \
            results require the declared double classes; authenticated automatic-residency inputs can gather under that context. Explicit \
            resident numeric options and exact native-integer controls require their declared RunMat extensions. Captures and inactive \
            operands are not recursively gathered. Finite host-memory policies are refused before provider, callback, file, pool, RNG or UI \
            effects because the complete solver/checkpoint workspace has no finite allocation bound. This is a host solver, not a GPU-native \
            or distributed algorithm.",
        ] },
    ],
    examples: examples::EXAMPLES,
    example_exemption: None, faqs: &[], media: &[],
    links: &[
        BuiltinDocumentationLink { label: "Surrogate optimization API", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/gads/surrogateopt.html") },
        BuiltinDocumentationLink { label: "Surrogate algorithm", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/gads/surrogate-optimization-algorithm.html") },
        BuiltinDocumentationLink { label: "Options and observers", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/gads/surrogate-optimization-options.html") },
        BuiltinDocumentationLink { label: "Checkpoint behavior", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/gads/work-with-checkpoint-files.html") },
    ],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Surrogate implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/optim/surrogateopt.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Measured history, numerical proposals and checkpoint lifecycle controls", location: "builtins::math::optim::surrogateopt::tests" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
