use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ridge"),
    slug: Some("ridge"),
    summary: "Fit ridge regression coefficients for one or more regularization parameters.",
    description: "`ridge(y,X,k)` estimates ridge-regression coefficients for a response vector, a predictor matrix, and one or more nonnegative regularization parameters.",
    keywords: &["ridge", "regression", "regularization", "statistics", "machine learning"],
    related: &["lasso"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Inputs and result shape",
            paragraphs: &[
                "`X` is an n-by-p real numeric or logical matrix and `y` is a vector with n elements. `k` is a nonempty scalar or vector of finite nonnegative numeric or logical values. Logical values enter the floating-point solve as zero or one. Each output column corresponds to one value of `k`, in input order.",
                "RunMat omits rows with NaN in `X` or `y` and rejects Inf. At least two complete observations are required. The documented floating input classes are single and double; typed-integer `X`, `y`, and `k` are separately gated RunMat extensions and must be exactly representable at the binary64 solver boundary.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Coefficient scaling",
            paragraphs: &[
                "By default (`scaled=1`), RunMat centers the response, centers and standardizes each predictor, and returns a p-by-numel(k) coefficient matrix without an intercept. Do not add a column of ones to `X` solely for an intercept.",
                "With `scaled=0`, the coefficients are restored to the original predictor scale and the first output row is the intercept; the output is (p+1)-by-numel(k). A typed-integer scaled flag is a separate RunMat extension and admits only exact zero or one.",
                "Fitting runs on the host and produces double coefficients. Admitted resident inputs are gathered at the host fitting boundary; there is no provider-native ridge solve or fusion route. MATLAB-compatible mode rejects the RunMat-only integer forms before provider access.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1", title: "Fit an original-scale ridge model",
            program: "X = [0; 1; 2; 3];\ny = [1; 3; 5; 7];\nB = ridge(y, X, 0, 0)",
            display_output: Some("B is approximately [1; 2]."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(B),[2 1])); assert(all(abs(B-[1;2])<1e-8));" },
        },
        BuiltinExample {
            id: "example-2", title: "Fit a ridge trace",
            program: "X = [0; 1; 2; 3];\ny = [1; 3; 5; 7];\nB = ridge(y, X, [0 1 10])",
            display_output: Some("B has one column per ridge parameter and contains standardized coefficients."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(B),[1 3])); assert(all(isfinite(B(:))));" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Should I include a column of ones in X?", answer: "No. The fit centers the response and predictors internally. Use scaled=0 to include the fitted intercept in the returned coefficients." },
        BuiltinDocumentationFaq { question: "How does scaled change the output size?", answer: "scaled=1 returns p rows. scaled=0 returns p+1 rows, with the intercept in the first row. Each column corresponds to one ridge parameter." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "lasso", target: BuiltinDocumentationLinkTarget::Builtin("lasso") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/ridge.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Ridge input admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/ridge/admission.rs") },
            BuiltinDocumentationLink { label: "Ridge preparation and solve", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/ridge/compute.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Ridge regression semantics", location: "builtins::stats::ml::ridge::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Statistics and machine-learning integration", location: "crates/runmat-vm/tests/stats_ml.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
