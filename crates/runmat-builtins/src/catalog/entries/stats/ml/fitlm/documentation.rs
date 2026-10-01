use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("fitlm"),
    slug: Some("fitlm"),
    summary: "Fit an ordinary least-squares linear regression model.",
    description: "`fitlm` fits an ordinary least-squares model from a numeric predictor matrix or table and returns a host `LinearModel` object. Single and double matrix data are supported in the MATLAB compatibility subset; RunMat also accepts logical predictor matrices and separately gates typed-integer predictors, responses, weights, selectors, and Intercept controls.",
    keywords: &["fitlm", "linear model", "regression", "least squares", "statistics", "machine learning"],
    related: &["predict", "lasso", "ridge"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Model forms",
            paragraphs: &[
                "`fitlm(X,y)` fits an intercept and one linear term per predictor column. Matrix model specifications `constant`, `linear`, `interactions`, `purequadratic`, and `quadratic` are supported.",
                "Table input uses the last variable as response by default, or takes a response name, additive formula such as `Y ~ A + B`, or `ResponseVar`. The fitted model exposes `Coefficients`, `Fitted`, `Residuals.Raw`, `NumObservations`, `DFE`, `RMSE`, `Rsquared`, `Formula`, `PredictorNames`, and `ResponseName`. The coefficient table has `Estimate`, `SE`, `tStat`, and `pValue` variables and coefficient row names.",
                "`Exclude` accepts logical masks or one-based numeric row indices; `Weights` accepts a numeric vector or, with a table, a weight variable name. Rows with NaN predictors or response, or zero weights, are omitted; Inf is rejected. Categorical predictors, robust regression, and Wilkinson interaction formula terms are explicitly unsupported.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Integer classes and residency",
            paragraphs: &[
                "All eight typed-integer classes are RunMat-only inputs for applicable roles. Predictor, response, and weight values must be exactly representable at the binary64 fitting boundary. Exclude indices and Intercept zero/nonzero controls are decoded from authoritative integer storage.",
                "Fitting runs on the host and returns a host model. Admitted resident inputs use the separately gated RunMat-only gather boundary; MATLAB-compatible mode rejects that extension before provider access. There is no device fitting kernel or fusion route.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1", title: "Fit a matrix-input linear model",
            program: "X = [0; 1; 2; 3];\ny = [1; 3; 5; 7];\nmdl = fitlm(X, y);\nmdl.Coefficients.Estimate",
            display_output: Some("The coefficient estimates are approximately [1; 2]."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(all(abs(mdl.Coefficients.Estimate - [1; 2]) < 1e-8));",
            },
        },
        BuiltinExample {
            id: "example-2", title: "Fit and predict",
            program: "X = [0; 1; 2; 3]; y = [1; 3; 5; 7];\nmdl = fitlm(X, y);\nyhat = predict(mdl, [4; 5])",
            display_output: Some("yhat is approximately [9; 11]."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(all(abs(yhat - [9; 11]) < 1e-8));",
            },
        },
        BuiltinExample {
            id: "example-3", title: "Fit a named table formula",
            program: "A = [0; 1; 0; 1]; B = [0; 0; 1; 1]; Y = 2 + 3*A + 4*B;\nT = table(A, B, Y, 'VariableNames', {'A','B','Y'});\nmdl = fitlm(T, 'Y ~ A + B')",
            display_output: Some("mdl uses A and B as predictors and Y as the response."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(mdl.NumPredictors == 2); assert(mdl.NumObservations == 4); assert(all(abs(mdl.Fitted - Y) < 1e-8));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does fitlm accept typed-integer predictors, responses, weights, or controls?",
            answer: "MATLAB-compatible mode rejects those typed-integer roles. RunMat mode separately gates predictors, responses, weights, Exclude/selectors, and Intercept. The fitted model remains floating and host-resident.",
        },
        BuiltinDocumentationFaq {
            question: "Does fitlm add an intercept?",
            answer: "Yes. The default model includes an intercept. Use the Intercept option or a no-intercept formula to omit it.",
        },
        BuiltinDocumentationFaq {
            question: "How are missing rows handled?",
            answer: "Rows with NaN in any active predictor or response are omitted before fitting.",
        },
    ],
    links: &[
        BuiltinDocumentationLink { label: "predict", target: BuiltinDocumentationLinkTarget::Builtin("predict") },
        BuiltinDocumentationLink { label: "lasso", target: BuiltinDocumentationLinkTarget::Builtin("lasso") },
        BuiltinDocumentationLink { label: "ridge", target: BuiltinDocumentationLinkTarget::Builtin("ridge") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/linear_model.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Argument admission and table forms", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/linear_model/admission.rs") },
            BuiltinDocumentationLink { label: "Preflight and typed option headers", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/linear_model/header.rs") },
            BuiltinDocumentationLink { label: "Option and formula parsing", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/linear_model/options.rs") },
            BuiltinDocumentationLink { label: "Numeric and integer value admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/linear_model/values.rs") },
            BuiltinDocumentationLink { label: "Host model fitting", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/linear_model/training.rs") },
            BuiltinDocumentationLink { label: "Model result construction", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/linear_model/projection.rs") },
            BuiltinDocumentationLink { label: "Prediction", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/linear_model/prediction.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Linear-model fitting and prediction", location: "builtins::stats::ml::linear_model::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "VM statistical models", location: "crates/runmat-vm/tests/stats_ml.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
