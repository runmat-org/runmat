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
                "Supported controls include MaxObjectiveEvaluations (1 through 2048), NumSeedPoints, AcquisitionFunctionName, ExplorationRatio, IsObjectiveDeterministic, Verbose, UseParallel=false, empty PlotFcn/OutputFcn and XConstraintFcn. Warm starts, conditional variables, nonempty plotting/output callbacks and parallel evaluation return explicit unsupported errors.",
                "The result exposes XTrace, ObjectiveTrace, ObjectiveMinimumTrace, ObjectiveEvaluationTimeTrace, FeasibilityTrace, OptimizationTrace, XAtMinObjective, MinObjective, BestPoint, BestObjective, VariableDescriptions, Options, Output, ExitFlag and surrogate metadata. Only the single-result form is currently implemented.",
                "This is a host control-flow solver. It materializes resident variable metadata and option inputs, invokes callbacks on the host, and returns a host result. It does not schedule surrogate updates or objective callbacks on a GPU provider; datastore-backed objective forms are not implemented.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1", title: "Minimize a scalar objective over two real variables",
            program: "x = optimizableVariable('x',[-1 1]); y = optimizableVariable('y',[-1 1]);\nobjective = @(X) X.x^2 + X.y^2;\nresults = bayesopt(objective, {x y}, 'MaxObjectiveEvaluations', 24, 'NumSeedPoints', 6);",
            display_output: Some("results contains trace tables and the best observed point."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(numel(results.ObjectiveTrace) > 0); assert(numel(results.ObjectiveTrace) <= 24);" },
        },
        BuiltinExample {
            id: "example-2", title: "Filter candidates with XConstraintFcn",
            program: "x = optimizableVariable('x',[-1 1]);\nobjective = @(X) X.x^2; isFeasible = @(X) X.x >= 0;\nresults = bayesopt(objective, x, 'XConstraintFcn', isFeasible, 'MaxObjectiveEvaluations', 8);",
            display_output: Some("Only feasible candidate rows are evaluated by the objective."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(numel(results.ObjectiveTrace) > 0); assert(numel(results.ObjectiveTrace) <= 8);" },
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
            BuiltinDocumentationLink { label: "Candidate search and Gaussian-process surrogate", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/bayesopt.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Bayesian optimization and callback tests", location: "builtins::stats::ml::bayesopt::tests" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
