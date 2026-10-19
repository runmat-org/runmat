use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("predict"),
    slug: Some("predict"),
    summary: "Predict responses from a fitted model.",
    description: "`predict(mdl,Xnew)` evaluates supported fitted `LinearModel`, `ClassificationTree`, `ClassificationLinear`, and deep-learning network objects on new predictor data.",
    keywords: &[
        "predict", "fitctree", "classification tree", "fitlm", "linear model", "dlnetwork",
        "regression", "prediction",
    ],
    related: &["dlnetwork", "forward", "fitctree", "fitlm"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Statistical models",
            paragraphs: &[
                "`Xnew` can be a numeric matrix with the predictor column count used during fitting. For table input, variables are selected by the model's `PredictorNames` property.",
                "`LinearModel` returns predicted responses. `[ypred,yci] = predict(...)` returns pointwise confidence intervals for curve predictions by default. `Alpha`, `Prediction`, and `Simultaneous` options are accepted; `Prediction` is `curve` or `observation`, and `Simultaneous` must be false because simultaneous intervals are not yet supported.",
                "`ClassificationTree` can return labels, posterior scores, one-based terminal node indices, and one-based class numbers. Tree prediction accepts matrix or table predictors and reports unsupported pruning or subtree options explicitly. `ClassificationLinear` can return labels and scores.",
                "Typed-integer predictors and numeric controls for implemented statistical models are independently gated RunMat-mode extensions with checked double boundaries.",
                "Implemented statistical predictors gather to host for model evaluation.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Deep-learning networks",
            paragraphs: &[
                "Deep-learning prediction uses the same admitted sequential layer execution as `forward`. A formatted dlarray input returns a formatted dlarray in the output axis order; an unformatted dlarray returns an unformatted dlarray; raw numeric input returns a numeric score array.",
                "For an unformatted dlarray, same-rank output retains the input axis order. If rank changes, batch precedes the remaining output axes. Raw numeric predictions use observation-major presentation.",
                "`ObservationsIn` accepts `rows` or `columns`. A formatted dlarray with incompatible labels is rejected instead of relabeled; the input dlarray and differentiation provenance are preserved.",
                "A dlnetwork can return updated State as a second output. Current SeriesNetwork and DAGNetwork compatibility routes return one output. Supported single- and double-precision resident GPU network inputs execute without gathering; unsupported explicit gpuArray routes return an error.",
                "Supported deep-learning networks retain their documented integer-input contract, separate from the statistical-model integer extensions.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1",
            title: "Predict from a linear model",
            program: "mdl = fitlm([0; 1; 2; 3], [1; 3; 5; 7]);\nyhat = predict(mdl, [4; 5])",
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
            id: "example-2",
            title: "Return confidence intervals",
            program: "mdl = fitlm([0; 1; 2; 3], [1; 3; 5; 7]);\n[yhat, yci] = predict(mdl, [4; 5], 'Alpha', 0.01)",
            display_output: Some("yci is a two-column matrix with lower and upper bounds."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(yci), [2 2])); assert(all(yhat >= yci(:,1) & yhat <= yci(:,2)));",
            },
        },
        BuiltinExample {
            id: "example-3",
            title: "Predict classes from a classification tree",
            program: "tree = fitctree([0; 1; 2; 3], [0; 0; 1; 1], 'MaxNumSplits', 1, 'MinParentSize', 2);\n[label,score] = predict(tree, [0.5; 2.5])",
            display_output: Some("label is [0; 1], and score has one posterior-probability column per class."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(label, [0; 1])); assert(isequal(size(score), [2 2]));",
            },
        },
        BuiltinExample {
            id: "example-4",
            title: "Predict from a dlnetwork",
            program: "layers = {featureInputLayer(2,'Name','in'); fullyConnectedLayer(2,'Name','fc'); softmaxLayer('Name','prob')};\nnet = dlnetwork(layers);\nscores = predict(net, [1 2; 3 4])",
            display_output: Some("scores has one row per observation."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(scores), [2 2])); assert(all(abs(sum(scores,2) - 1) < 1e-8));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Which model classes are supported?",
        answer: "Currently `predict` supports `LinearModel` from `fitlm`, `ClassificationTree` from `fitctree`, `ClassificationLinear` from `fitclinear`, and supported deep-learning network objects.",
    }],
    links: &[
        BuiltinDocumentationLink { label: "dlnetwork", target: BuiltinDocumentationLinkTarget::Builtin("dlnetwork") },
        BuiltinDocumentationLink { label: "forward", target: BuiltinDocumentationLinkTarget::Builtin("forward") },
        BuiltinDocumentationLink { label: "fitctree", target: BuiltinDocumentationLinkTarget::Builtin("fitctree") },
        BuiltinDocumentationLink { label: "fitlm", target: BuiltinDocumentationLinkTarget::Builtin("fitlm") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/linear_model.rs") },
        BuiltinDocumentationLink { label: "Prediction dispatcher", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/predict.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Linear-model prediction and confidence intervals", target: BuiltinDocumentationLinkTarget::Source("crates/runmat-runtime/src/builtins/stats/ml/linear_model.rs") },
            BuiltinDocumentationLink { label: "Prediction dispatch", target: BuiltinDocumentationLinkTarget::Source("crates/runmat-runtime/src/builtins/stats/ml/predict.rs") },
            BuiltinDocumentationLink { label: "Deep-learning execution", target: BuiltinDocumentationLinkTarget::Source("crates/runmat-runtime/src/builtins/deep_learning/model/execution/network.rs") },
            BuiltinDocumentationLink { label: "Deep-learning output presentation", target: BuiltinDocumentationLinkTarget::Source("crates/runmat-runtime/src/builtins/deep_learning/model/output.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Statistical model prediction", location: "builtins::stats::ml::linear_model::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Deep-learning network prediction", location: "builtins::deep_learning::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM statistics integration", location: "crates/runmat-vm/tests/stats_ml.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
