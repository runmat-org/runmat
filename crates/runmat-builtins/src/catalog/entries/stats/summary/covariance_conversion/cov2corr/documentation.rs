use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("cov2corr"),
    slug: Some("cov2corr"),
    summary: "Convert expected covariance to standard deviations and a correlation matrix.",
    description: "`ExpSigma = cov2corr(ExpCovariance)` returns a row of standard deviations. `[ExpSigma,ExpCorrC] = cov2corr(ExpCovariance)` also returns the correlation matrix. This follows the Finance Toolbox output order, which differs from `corrcov`.",
    keywords: &["cov2corr", "covariance", "correlation", "standard deviation", "statistics"],
    related: &["corrcov", "cov", "corrcoef"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`ExpCovariance` must be a finite, symmetric, positive-semidefinite square matrix. Variances must be nonnegative and covariances must satisfy their pairwise variance bounds.",
                "A scalar real covariance input is treated as a 1-by-1 matrix. Symmetry and covariance bounds use floating-point tolerance.",
                "The first output, `ExpSigma`, is a 1-by-n row vector of `sqrt(diag(ExpCovariance))`; the second, `ExpCorrC`, is the n-by-n correlation matrix. A zero-variance correlation denominator yields `NaN`.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Classes and execution",
            paragraphs: &[
                "The documented Finance Toolbox input is real double. RunMat mode additionally accepts real single, typed-integer, and logical covariance inputs. Single inputs produce single outputs; integer and logical inputs produce double outputs.",
                "Resident GPU input is a RunMat extension. An available provider hook can keep both outputs resident; unsupported hooks can use the documented host materialization path. The operation is a fusion boundary.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1",
            title: "Finance Toolbox output order",
            program: "C = [16 4; 4 25]; [ExpSigma,ExpCorrC] = cov2corr(C)",
            display_output: Some("ExpSigma = [4 5]\nExpCorrC = [1 0.2; 0.2 1]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(ExpSigma),[1 2])); assert(isequal(ExpSigma,[4 5])); assert(max(abs(ExpCorrC(:)-[1;0.2;0.2;1]))<1e-12);" },
        },
        BuiltinExample {
            id: "example-2",
            title: "Single output is the standard-deviation row",
            program: "C = [9 0; 0 16]; s = cov2corr(C)",
            display_output: Some("s = [3 4]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(s,[3 4]));" },
        },
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Is cov2corr an alias for corrcov?", answer: "No. `cov2corr` returns standard deviations first as a row; `corrcov` returns correlation first and its standard deviations as a column." }],
    links: &[
        BuiltinDocumentationLink { label: "corrcov", target: BuiltinDocumentationLinkTarget::Builtin("corrcov") },
        BuiltinDocumentationLink { label: "cov", target: BuiltinDocumentationLinkTarget::Builtin("cov") },
        BuiltinDocumentationLink { label: "corrcoef", target: BuiltinDocumentationLinkTarget::Builtin("corrcoef") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/covariance_conversion.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Covariance validation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/covariance_conversion/matrix_validation.rs") },
            BuiltinDocumentationLink { label: "Correlation computation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/covariance_conversion/computation.rs") },
            BuiltinDocumentationLink { label: "Provider dispatch and admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/summary/covariance_conversion/provider.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Covariance conversion output order", location: "crates/runmat-runtime/src/builtins/stats/summary/covariance_conversion/tests.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Resident conversion ownership and errors", location: "crates/runmat-runtime/src/builtins/stats/summary/covariance_conversion/provider/tests.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled covariance conversion", location: "crates/runmat-vm/tests/stats_ml.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
