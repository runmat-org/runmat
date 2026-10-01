use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("fitctree"),
    slug: Some("fitctree"),
    summary: "Fit a binary-split classification tree for multiclass labels.",
    description: "`fitctree` fits a numeric-predictor classification tree from a matrix or table and returns a `ClassificationTree` object usable with `predict`.",
    keywords: &["fitctree", "classification tree", "decision tree", "machine learning", "statistics"],
    related: &["predict", "fitlm"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Inputs and class labels",
            paragraphs: &[
                "`fitctree(X,Y)` fits a tree with binary splits, observations in rows, and class labels in `Y`. Table forms take a response-variable name, an additive formula such as `Y ~ A + B`, or an external response vector containing one label per table row.",
                "Labels may be numeric, logical, string, char, cell-string, or any of the eight signed and unsigned integer classes. ClassNames and prediction retain exact integer identity above flintmax.",
                "Typed-integer matrix and active table predictors are separately gated RunMat extensions and must be exactly representable in binary64 at the host fitting boundary. Typed-integer weights and numerical controls use another extension gate. MATLAB-compatible mode rejects those extensions while accepting the documented floating forms.",
                "Rows with missing response, zero weight, or all missing predictors are omitted; partially missing predictor rows remain eligible for splits on observed variables. Inf and negative weights are rejected.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Options and result",
            paragraphs: &[
                "Supported name-value options include ClassNames, PredictorNames, ResponseName, MaxNumSplits, MinLeafSize, MinParentSize, SplitCriterion, Weights, and ScoreTransform. SplitCriterion accepts `gdi` and `deviance`; ScoreTransform currently accepts only `none`.",
                "The model exposes ResponseName, PredictorNames, ClassNames, CategoricalPredictors, ScoreTransform, NumObservations, NumPredictors, NumNodes, NumSplits, and ModelParameters. Categorical predictors, pruning, surrogate splits, cross-validation, prior/cost matrices, and hyperparameter optimization are explicitly unsupported.",
                "Fitting runs on CPU and returns a host model. Provider-resident inputs use an independently gated RunMat-only gather boundary; MATLAB-compatible mode rejects this before provider access. There is no tree-fitting device kernel or fusion route.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1", title: "Fit and predict numeric classes",
            program: "X = [0; 1; 2; 3];\nY = [0; 0; 1; 1];\nmdl = fitctree(X, Y, 'MaxNumSplits', 1, 'MinParentSize', 2);\n[label,score] = predict(mdl, [0.5; 2.5])",
            display_output: Some("label is [0; 1], and score has one posterior-probability column per class."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(label,[0;1])); assert(isequal(size(score),[2 2])); assert(all(abs(sum(score,2)-1)<1e-8));",
            },
        },
        BuiltinExample {
            id: "example-2", title: "Fit a table-input classification tree",
            program: "A = [0; 0; 1; 1]; B = [0; 1; 0; 1]; Y = [0; 0; 1; 1];\nT = table(A, B, Y, 'VariableNames', {'A','B','Y'});\nmdl = fitctree(T, 'Y ~ A + B')",
            display_output: Some("mdl uses A and B as numeric predictors and Y as the response."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(mdl.NumPredictors == 2); assert(mdl.NumObservations == 4);",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does fitctree support categorical predictors?",
            answer: "No. This implementation supports numeric predictors and returns an explicit error for categorical-predictor options.",
        },
        BuiltinDocumentationFaq {
            question: "What does predict return for a classification tree?",
            answer: "One output returns labels. With multiple outputs, [label,score,node,cnum] supplies posterior scores, one-based leaf ids, and one-based class numbers.",
        },
    ],
    links: &[
        BuiltinDocumentationLink { label: "predict", target: BuiltinDocumentationLinkTarget::Builtin("predict") },
        BuiltinDocumentationLink { label: "fitlm", target: BuiltinDocumentationLinkTarget::Builtin("fitlm") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/classification_tree.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Argument admission and table forms", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/classification_tree/admission.rs") },
            BuiltinDocumentationLink { label: "Preflight and typed option headers", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/classification_tree/header.rs") },
            BuiltinDocumentationLink { label: "Option parsing", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/classification_tree/options.rs") },
            BuiltinDocumentationLink { label: "Numeric and integer value admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/classification_tree/values.rs") },
            BuiltinDocumentationLink { label: "Binary-split training", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/classification_tree/training.rs") },
            BuiltinDocumentationLink { label: "Exact label representation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/classification_tree/labels.rs") },
            BuiltinDocumentationLink { label: "Model result construction", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/classification_tree/projection.rs") },
            BuiltinDocumentationLink { label: "Prediction", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/classification_tree/prediction.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Classification-tree fitting and prediction", location: "builtins::stats::ml::classification_tree::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "VM statistical models", location: "crates/runmat-vm/tests/stats_ml.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
