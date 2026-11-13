use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Discriminant implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/classification.rs"),
};

const ADMISSION: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Predictor and label admission",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/classification/header.rs"),
};
const LABELS: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Exact label identities and output families",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/classification/labels.rs"),
};
const PRIORS: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Prior admission and normalization",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/classification/prior.rs"),
};
const MODEL: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Discriminant model preparation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/classification/model.rs"),
};
const PREDICTION: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Prediction and requested numerical outputs",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/classification/prediction.rs"),
};
const COEFFICIENTS: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Pairwise discriminant coefficients",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/classification/coefficients.rs"),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("classify"), slug: Some("classify"),
    summary: "Classify observations using discriminant analysis.",
    description: "`classify(sample,training,group)` fits a discriminant model to training rows and assigns one label to each sample row. The model is used for this call rather than returned as a reusable object.",
    keywords: &["classify", "discriminant analysis", "lda", "qda", "classification", "statistics", "machine learning"],
    related: &["fitctree", "predict", "cvpartition"],
    sections: &[
        BuiltinDocumentationSection { heading: "Predictors, labels, and priors", paragraphs: &[
            "`sample` and `training` are finite real numeric matrices with matching columns and observations in rows. `group` has one label per training row; numeric, logical, string, character-row, and cell-string labels are admitted. Missing numeric, empty text, and typed missing string labels are excluded from fitting. Literal `<missing>` text is an ordinary label. Predicted labels retain the input label family and numeric class, including exact wide integers.",
            "The default is linear discriminant analysis with equal priors. The `type` control accepts `linear`, `quadratic`, `diagLinear`, `diagQuadratic`, and `mahalanobis`. `prior` is a numeric vector, `empirical`, or a struct with group and prob fields. Mahalanobis classification ignores prior values for assignments and returns NaN posterior/log-density values.",
            "Typed-integer predictors and prior probabilities are separately gated RunMat extensions and cross a checked exact-binary64 statistical boundary. Typed-integer labels remain exact and may exceed `flintmax`. Logical predictors and resident input are separate RunMat-only extensions; logical priors are rejected.",
        ] },
        BuiltinDocumentationSection { heading: "Outputs and execution", paragraphs: &[
            "Request `[class,err,posterior,logp,coeff]` for predicted labels, prior-weighted apparent training error, posterior probabilities, unconditional log densities, and pairwise discriminant coefficients. `coeff` is a k-by-k structure array indexed as `coeff(i,j)`, with type, class-name, constant, linear, and where applicable quadratic fields for the boundary between classes i and j.",
            "Resident inputs are recursively gathered after compatibility admission; fitting and prediction run on the host. No provider-resident discriminant kernel or fusion route is claimed.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "numeric", title: "Classify two numeric observations",
            program: "training = [0; 0.5; 2; 2.5]; group = [1; 1; 2; 2]; [class,err,posterior] = classify([0.2; 2.2], training, group); disp(class); disp(err);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(class,[1;2])); assert(err == 0); assert(isequal(size(posterior),[2 2]));" },
        },
        BuiltinExample { id: "text-prior", title: "Use text labels and empirical priors",
            program: "group = [\"low\"; \"low\"; \"high\"; \"high\"]; class = classify([0.1; 3.2], [0; 0.4; 3; 3.4], group, 'quadratic', 'empirical'); disp(class);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isstring(class)); assert(isequal(class,[\"low\";\"high\"]));" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does classify train a reusable model object?", answer: "No. It fits and predicts in one call. Use a model-fitting helper such as `fitctree` for a reusable object." },
        BuiltinDocumentationFaq { question: "Are priors used for mahalanobis classification?", answer: "No. Prior values are admitted for compatibility but do not change the Mahalanobis distance assignment." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "fitctree", target: BuiltinDocumentationLinkTarget::Builtin("fitctree") },
        BuiltinDocumentationLink { label: "predict", target: BuiltinDocumentationLinkTarget::Builtin("predict") },
        BuiltinDocumentationLink { label: "cvpartition", target: BuiltinDocumentationLinkTarget::Builtin("cvpartition") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE, ADMISSION, LABELS, PRIORS, MODEL, PREDICTION, COEFFICIENTS],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Requested outputs, priors, and discriminants", location: "crates/runmat-runtime/src/builtins/stats/ml/classification/tests/core.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Label families", location: "crates/runmat-runtime/src/builtins/stats/ml/classification/tests/labels.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Exact integer roles", location: "crates/runmat-runtime/src/builtins/stats/ml/classification/tests/integer.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Preflight, provider failures, and borrowed allocations", location: "crates/runmat-runtime/src/builtins/stats/ml/classification/tests/provider.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Classification from MATLAB-syntax scripts", location: "crates/runmat-vm/tests/stats_ml.rs" },
        ],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
