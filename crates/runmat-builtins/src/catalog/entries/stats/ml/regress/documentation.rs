use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("regress"),
    slug: Some("regress"),
    summary: "Fit a multiple linear regression model by ordinary least squares.",
    description: "`regress(y,X)` estimates coefficients for the linear model `y = X*b`. Include a column of ones in `X` when an intercept is required.",
    keywords: &["regress", "linear regression", "least squares", "statistics", "machine learning"],
    related: &["fitlm", "ridge", "lasso"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Inputs and computation",
            paragraphs: &[
                "`y` is a numeric response vector with one value per row of the two-dimensional design matrix `X`. The documented floating computation admits single and double. Typed-integer y, X, and alpha are RunMat extensions: each integer must cross the checked exact binary64 boundary, and MATLAB compatibility mode rejects the extension before provider access. Logical y and X are also admitted by the current host conversion.",
                "Rows with `NaN` in y or any predictor are omitted. An `Inf` in y or X is an error. The optional significance level alpha defaults to 0.05 and must be a scalar in [0,1). A rank-deficient design uses rank-revealing column selection; dependent coefficients and their intervals are zero.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Outputs and execution",
            paragraphs: &[
                "The requested outputs are `[b,bint,r,rint,stats]`, including partial forms. Coefficient intervals are in bint. Residuals and their studentized, deleted-variance intervals are in r and rint. The 1-by-4 stats row contains `[R2 F p errorVariance]`. Interval values can be NaN when the available degrees of freedom are insufficient.",
                "Admitted GPU inputs are gathered to host. The SVD-backed coefficient fit and interval diagnostics run on CPU; there is no provider-native regression kernel, GPU fusion, or implicit intercept term.",
            ],
        },
    ],
    examples: &[BuiltinExample {
        id: "example-1",
        title: "Fit a line with an explicit intercept column",
        program: "X = [ones(4,1) (0:3)']; y = [1;3;5;7]; [b,bint,r,rint,stats] = regress(y,X)",
        display_output: Some("b is approximately [1; 2] and stats(1) is 1."),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(b),[2 1])); assert(all(abs(b-[1;2])<1e-8)); assert(isequal(size(bint),[2 2])); assert(isequal(size(r),[4 1])); assert(isequal(size(rint),[4 2])); assert(isequal(size(stats),[1 4])); assert(abs(stats(1)-1)<1e-8);",
        },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Does regress add an intercept automatically?",
        answer: "No. Add a column of ones to X when the model should include an intercept.",
    }],
    links: &[
        BuiltinDocumentationLink { label: "fitlm", target: BuiltinDocumentationLinkTarget::Builtin("fitlm") },
        BuiltinDocumentationLink { label: "ridge", target: BuiltinDocumentationLinkTarget::Builtin("ridge") },
        BuiltinDocumentationLink { label: "lasso", target: BuiltinDocumentationLinkTarget::Builtin("lasso") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/regress.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Input admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/regress/admission.rs") },
            BuiltinDocumentationLink { label: "Least-squares fit", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/regress/fit.rs") },
            BuiltinDocumentationLink { label: "Rank selection", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/regress/rank.rs") },
            BuiltinDocumentationLink { label: "SVD solve", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/regress/solver.rs") },
            BuiltinDocumentationLink { label: "Interval diagnostics", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/regress/diagnostics.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Regression fit and diagnostics", location: "builtins::stats::ml::regress::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM regression forms", location: "crates/runmat-vm/tests/stats_ml.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
