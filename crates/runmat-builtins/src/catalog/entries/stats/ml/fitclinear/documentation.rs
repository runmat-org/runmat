use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("fitclinear"),
    slug: Some("fitclinear"),
    summary: "Fit a binary linear classification model.",
    description: "`fitclinear` fits a binary linear classifier from a predictor matrix or table and returns a `ClassificationLinear` object usable with `predict`.",
    keywords: &["fitclinear", "classification linear", "svm", "logistic regression", "machine learning", "statistics"],
    related: &["predict", "fitctree"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Observations and training",
            paragraphs: &[
                "`fitclinear(X,Y)` fits a binary SVM-style linear classifier with observations in rows. Table input accepts a response name, an additive formula such as `Y ~ A + B`, or an external response vector.",
                "Learner accepts `svm` and `logistic`. Logistic training uses logistic loss and returns posterior-style scores by default; SVM training returns signed-margin scores. Regularization accepts `ridge` and `lasso`. Lambda accepts a nonnegative scalar or vector, producing one coefficient column per Lambda value.",
                "ObservationsIn accepts `rows` or `columns` for numeric matrices; tables require row observations. Operative name-value controls include ClassNames, PredictorNames, ResponseName, Weights, Prior, FitBias, Beta, Bias, Solver, IterationLimit, PassLimit, and ScoreTransform. BatchLimit, BetaTolerance, DeltaGradientTolerance, GradientTolerance, NumCheckConvergence, PostFitBias, and Verbose are accepted for script compatibility but do not alter this solver. Cost matrices, cross-validation, partitions, and hyperparameter optimization are explicitly unsupported.",
                "Rows with NaN predictors or response, or zero weights, are omitted. Inf predictors and negative weights are rejected. The result exposes Beta, Bias, Lambda, Learner, Regularization, Solver, ScoreTransform, ResponseName, PredictorNames, ClassNames, Prior, NumObservations, NumPredictors, and ModelParameters. A second output FitInfo records Lambda, Objective, NumIterations, NumPasses, and GradientNorm.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Label classes, integers and residency",
            paragraphs: &[
                "Binary labels may be numeric, logical, string, char, cell-string, or any of the eight fixed-width integer classes. Integer grouping identity and class remain exact through ClassNames and prediction, including adjacent 64-bit labels above flintmax.",
                "Typed-integer predictor values must be exactly representable in binary64 at the host statistical computation boundary. Typed-integer Weights and numeric controls have a separate extension gate and the same exactness check. MATLAB-compatible mode rejects the RunMat-only integer forms.",
                "Fitting runs on CPU and returns a host model. Provider-resident inputs use an independently gated RunMat-only gather boundary before fitting; MATLAB-compatible mode rejects this before provider access. There is no device fitting kernel or fusion route.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1", title: "Fit and predict a logistic linear classifier",
            program: "X = [0; 1; 2; 3];\nY = [0; 0; 1; 1];\n[Mdl,FitInfo] = fitclinear(X, Y, 'Learner', 'logistic', 'Lambda', 0);\n[label,score] = predict(Mdl, [0.2; 2.8])",
            display_output: Some("label is [0; 1], and score contains one probability column per class."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(label,[0;1])); assert(isequal(size(score),[2 2])); assert(all(score(:)>=0 & score(:)<=1)); assert(numel(FitInfo.Lambda)==1);",
            },
        },
        BuiltinExample {
            id: "example-2", title: "Use column-oriented observations",
            program: "Y = [0; 0; 1; 1];\nMdl = fitclinear([0 1 2 3], Y, 'ObservationsIn', 'columns');\nlabel = predict(Mdl, [0.2 2.8], 'ObservationsIn', 'columns')",
            display_output: Some("The predictor matrix is transposed internally so columns are observations."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(label,[0;1])); assert(Mdl.NumObservations==4);",
            },
        },
        BuiltinExample {
            id: "example-3", title: "Fit a table-input classifier",
            program: "A = [0; 0; 1; 1]; B = [0; 1; 0; 1]; Y = [0; 0; 1; 1];\nT = table(A, B, Y, 'VariableNames', {'A','B','Y'});\nMdl = fitclinear(T, 'Y ~ A + B', 'Learner', 'svm')",
            display_output: Some("Mdl uses A and B as numeric predictors and Y as the binary response."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(Mdl.NumPredictors==2); assert(Mdl.NumObservations==4);",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does fitclinear support multiclass classification?",
            answer: "No. It requires exactly two observed classes and reports an error otherwise.",
        },
        BuiltinDocumentationFaq {
            question: "What does predict return for ClassificationLinear?",
            answer: "One output gives class labels. A second output gives signed-margin scores for SVM models or posterior-style probabilities for logistic models.",
        },
    ],
    links: &[
        BuiltinDocumentationLink { label: "predict", target: BuiltinDocumentationLinkTarget::Builtin("predict") },
        BuiltinDocumentationLink { label: "fitctree", target: BuiltinDocumentationLinkTarget::Builtin("fitctree") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/classification_linear.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Argument admission and table forms", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/classification_linear/admission.rs") },
            BuiltinDocumentationLink { label: "Preflight and typed option headers", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/classification_linear/header.rs") },
            BuiltinDocumentationLink { label: "Option parsing and ignored controls", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/classification_linear/options.rs") },
            BuiltinDocumentationLink { label: "Numeric and integer value admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/classification_linear/values.rs") },
            BuiltinDocumentationLink { label: "Host classifier training", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/classification_linear/training.rs") },
            BuiltinDocumentationLink { label: "Exact label representation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/classification_linear/labels.rs") },
            BuiltinDocumentationLink { label: "FitInfo result", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/classification_linear/projection.rs") },
            BuiltinDocumentationLink { label: "Prediction", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/classification_linear/prediction.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Linear-classifier fitting and prediction", location: "builtins::stats::ml::classification_linear::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "VM statistical models", location: "crates/runmat-vm/tests/stats_ml.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
