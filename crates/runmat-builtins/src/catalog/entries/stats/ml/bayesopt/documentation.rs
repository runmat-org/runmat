use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("bayesopt"),
    slug: Some("bayesopt"),
    summary: "Run deterministic Bayesian optimization over optimizableVariable metadata.",
    description: "`bayesopt(fun, vars, Name, Value)` evaluates a scalar objective over `optimizableVariable` metadata and returns a host `BayesianOptimization` result object.",
    keywords: &["bayesopt", "Bayesian optimization", "optimizableVariable", "expected improvement"],
    related: &["optimizableVariable"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Variables, callbacks and search",
            paragraphs: &[
                "The objective is invoked as fun(X), where X is a one-row table with a column for each variable. Real, integer and categorical variables are supported, including valid logarithmic numeric ranges and Optimize=false fixed variables.",
                "The in-memory search uses deterministic low-discrepancy seed points, then a Gaussian-process surrogate with expected-improvement-style acquisition. It stops when its finite unique candidate pool is exhausted, even if MaxObjectiveEvaluations is larger; the pool is not an exhaustive enumeration of every possible point in a large domain.",
                "The objective can return a scalar value or a struct with Objective and optional ConstraintViolations. Violations at or below zero are feasible. XConstraintFcn, when supplied, is a logical feasibility filter evaluated before the objective.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Options, results and limits",
            paragraphs: &[
                "Search controls include MaxObjectiveEvaluations (1 through 2048, default 30), NumSeedPoints (0 through 2048, default 4, limited to MaxObjectiveEvaluations), AcquisitionFunctionName and ExplorationRatio. Acquisition names are expected-improvement, expected-improvement-plus (default) and lower-confidence-bound. The per-second acquisition spellings are accepted aliases of their corresponding expected-improvement variants; the current search does not divide acquisition scores by evaluation time.",
                "IsObjectiveDeterministic and Verbose are accepted and recorded in Options, but currently do not change the search or emit progress output. UseParallel=false, empty PlotFcn/OutputFcn and XConstraintFcn are supported. Warm starts, conditional variables, nonempty plotting/output callbacks and parallel evaluation return explicit unsupported errors.",
                "The result exposes XTrace, ObjectiveTrace, ObjectiveMinimumTrace, ObjectiveEvaluationTimeTrace, FeasibilityTrace, OptimizationTrace, XAtMinObjective, MinObjective, BestPoint, BestObjective, VariableDescriptions, Options, Output, ExitFlag and surrogate metadata. A call can return one result or run without retaining a result; requesting more than one output is unsupported.",
                "This is a host control-flow solver. It validates host-visible controls before downloading resident option values and variable metadata, invokes callbacks on the host, and returns a host result. It does not schedule surrogate updates or objective callbacks on a GPU provider; datastore-backed objective forms are not implemented.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1", title: "Minimize a scalar objective over two real variables",
            program: "x = optimizableVariable('x',[-1 1]); y = optimizableVariable('y',[-1 1]);\nobjective = @(X) X.x^2 + X.y^2;\nresults = bayesopt(objective, {x y}, 'MaxObjectiveEvaluations', 24, 'NumSeedPoints', 6);\ndisp(results);",
            display_output: Some("results contains trace tables and the best observed point."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(numel(results.ObjectiveTrace) > 0); assert(numel(results.ObjectiveTrace) <= 24); assert(istable(results.XTrace)); assert(abs(results.MinObjective - min(results.ObjectiveTrace)) < 1e-12); assert(abs(results.XAtMinObjective.x^2 + results.XAtMinObjective.y^2 - results.MinObjective) < 1e-12);" },
        },
        BuiltinExample {
            id: "example-2", title: "Filter candidates with XConstraintFcn",
            program: "x = optimizableVariable('x',[-1 1]);\nobjective = @(X) X.x^2; isFeasible = @(X) X.x >= 0;\nresults = bayesopt(objective, x, 'XConstraintFcn', isFeasible, 'MaxObjectiveEvaluations', 8);\ndisp(results);",
            display_output: Some("Only feasible candidate rows are evaluated by the objective."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(numel(results.ObjectiveTrace) > 0); assert(numel(results.ObjectiveTrace) <= 8); assert(all(results.XTrace.x >= 0)); assert(abs(results.MinObjective - min(results.ObjectiveTrace)) < 1e-12);" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Can bayesopt run on a GPU?", answer: "The optimization loop and callbacks run on the host. Resident metadata/options are gathered before evaluation; there is no bayesopt GPU kernel." },
        BuiltinDocumentationFaq { question: "Is the search stochastic?", answer: "The current in-memory candidate sequence is deterministic, including its seed points and Gaussian-process acquisition." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "optimizableVariable", target: BuiltinDocumentationLinkTarget::Builtin("optimizableVariable") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/bayesopt.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Variable and option admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/bayesopt.rs") },
            BuiltinDocumentationLink { label: "Admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/bayesopt/admission.rs") },
            BuiltinDocumentationLink { label: "Options", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/bayesopt/options.rs") },
            BuiltinDocumentationLink { label: "Variable decoding", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/bayesopt/variables.rs") },
            BuiltinDocumentationLink { label: "Optimization loop", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/bayesopt/optimizer.rs") },
            BuiltinDocumentationLink { label: "Candidate search", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/bayesopt/candidate.rs") },
            BuiltinDocumentationLink { label: "Gaussian-process surrogate", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/bayesopt/surrogate.rs") },
            BuiltinDocumentationLink { label: "Callbacks", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/bayesopt/callbacks.rs") },
            BuiltinDocumentationLink { label: "Results", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/bayesopt/result.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Bayesian optimization and callback tests", location: "builtins::stats::ml::bayesopt::tests" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
