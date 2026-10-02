use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("perfcurve"),
    slug: Some("perfcurve"),
    summary: "Compute classifier performance curves and AUC values.",
    description: "`perfcurve(labels,scores,posclass)` treats `posclass` as the positive class, all other nonmissing classes as negative classes, and returns ROC false-positive-rate values. `[X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(...)` returns the Y criterion, thresholds, trapezoidal AUC, optimal ROC point, per-negative-subclass Y values, and subclass labels as a cell array.",
    keywords: &["perfcurve", "roc", "auc", "classification", "statistics", "machine learning"],
    related: &["confusionmat", "classify", "fitctree"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Labels, scores, and missing rows",
            paragraphs: &[
                "Labels may be numeric, logical, string, character rows, categorical values, or cell strings. Text label families can be mixed between string, char, cell-string, and categorical inputs. Typed-integer labels are a RunMat-mode extension and remain exact during class comparison, including adjacent 64-bit values above `flintmax`; typed-integer scores and numeric options must be exactly representable in double before entering curve computation.",
                "`scores` must be a numeric vector with one score per label. Score matrices for class-column inference are rejected explicitly until the classifier object/class-order surface is available. Rows with missing labels, zero weights, or `NaN` scores are omitted by default. `\"ProcessNaN\",\"addtofalse\"` keeps `NaN` scores: positive-class `NaN` scores are false negatives and negative-class `NaN` scores are false positives at every threshold.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Criteria, thresholds, and outputs",
            paragraphs: &[
                "Default thresholds include an initial reject-all point whose threshold duplicates the highest score, followed by each unique finite score in descending order. `\"TVals\",t` uses explicit thresholds and cannot be combined with `\"XVals\"`. `\"XCrit\"` and `\"YCrit\"` accept `fpr`, `tpr`/`sens`/`reca`, `tnr`/`spec`, `fnr`/`miss`, `ppv`/`prec`, `npv`, `accu`, `tp`, `fn`, `fp`, `tn`, `tp+fp`, `rpp`, `rnp`, and `ecost`.",
                "`\"NegClass\",labels` restricts negative classes. `SUBY` contains one Y-criterion column per selected negative subclass and `SUBYNAMES` returns the corresponding labels as an n-by-1 cell array. `\"XVals\",x` returns curve values at requested X positions and cannot be combined with `TVals`. `UseNearest` defaults to true: nearest mode returns actual X values in requested order; false sorts requested X values and linearly interpolates finite X/Y points. AUC is computed over the full curve interval covered by `XVals`.",
                "`OPTROCPT` is `[NaN NaN]` for non-ROC criteria. Confidence-bound and bootstrap modes such as `NBoot` are not implemented and return a dedicated unsupported-mode error. A call that requests no outputs still checks compatibility, integer exactness, and host-visible argument and option forms, then returns without downloading resident inputs or computing the curve; finite-score and other curve-specific data checks are not performed in that case.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Weights, prior, cost, and execution",
            paragraphs: &[
                "`\"Weights\",w` applies nonnegative observation weights. `Prior` accepts `empirical`, `uniform`, or a two-element positive/negative prior vector and rescales class weights. `Cost` accepts a nonnegative 2-by-2 matrix and is used for `ecost` and standard ROC optimal-point selection.",
                "Admitted gpuArray inputs are materialized to host. Label handling, threshold sweeps, and AUC calculation run on CPU; there is no provider-native perfcurve reduction or fusion. Runtime-owned gathering may suspend. The output curves are host double arrays and subclass names are a host cell array.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "roc-auc",
            title: "Compute a basic ROC curve and AUC",
            program: "labels = [1; 0; 1; 0]; scores = [0.9; 0.8; 0.4; 0.1]; [X,Y,T,AUC] = perfcurve(labels, scores, 1)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(abs(AUC-0.75)<1e-12); assert(isequal(size(X),size(Y))); assert(isequal(size(X),size(T)));" },
        },
        BuiltinExample {
            id: "precision-recall",
            title: "Compute precision-recall values at explicit thresholds",
            program: "labels = [1;0;1;0]; scores = [0.9;0.8;0.4;0.1]; [R,P] = perfcurve(labels, scores, 1, \"XCrit\", \"reca\", \"YCrit\", \"prec\", \"TVals\", [0.8 0.4])",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(R),[2 1])); assert(isequal(size(P),[2 1])); assert(max(abs(R-[0.5;1]))<1e-12); assert(max(abs(P-[0.5;2/3]))<1e-12);" },
        },
        BuiltinExample {
            id: "negative-subclass",
            title: "Restrict negative classes",
            program: "labels = [\"hit\";\"miss\";\"hit\";\"other\"]; scores = [0.9;0.8;0.4;0.1]; [X,Y,T,AUC,opt,suby,names] = perfcurve(labels, scores, \"hit\", \"NegClass\", \"miss\")",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(suby,2),1)); assert(numel(names)==1); assert(isequal(size(opt),[1 2]));" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does perfcurve train a classifier?", answer: "No. It evaluates existing classifier scores against known labels. Train or evaluate a model separately and pass the score vector for the positive class." },
        BuiltinDocumentationFaq { question: "Why are confidence intervals not returned?", answer: "RunMat implements deterministic curve, threshold, AUC, optimal-point, weighting, prior, cost, and subclass outputs. Bootstrap confidence-bound modes require separate statistical resampling and are rejected explicitly." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "confusionmat", target: BuiltinDocumentationLinkTarget::Builtin("confusionmat") },
        BuiltinDocumentationLink { label: "classify", target: BuiltinDocumentationLinkTarget::Builtin("classify") },
        BuiltinDocumentationLink { label: "fitctree", target: BuiltinDocumentationLinkTarget::Builtin("fitctree") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/perfcurve.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Typed admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/perfcurve/admission.rs") },
            BuiltinDocumentationLink { label: "Class identity", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/perfcurve/labels.rs") },
            BuiltinDocumentationLink { label: "Observation weights and prior", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/perfcurve/curve/observations.rs") },
            BuiltinDocumentationLink { label: "Threshold and count core", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/perfcurve/curve/core.rs") },
            BuiltinDocumentationLink { label: "Interpolation and AUC", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/perfcurve/curve/interpolation.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Performance curve semantics", location: "crates/runmat-runtime/src/builtins/stats/ml/perfcurve/tests.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM statistics/ML forms", location: "crates/runmat-vm/tests/stats_ml.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
