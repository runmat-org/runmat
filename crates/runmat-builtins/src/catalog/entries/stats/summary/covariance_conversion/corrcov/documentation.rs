use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("corrcov"),
    slug: Some("corrcov"),
    summary: "Convert a covariance matrix to correlation coefficients and standard deviations.",
    description: "`R = corrcov(C)` normalizes a real covariance matrix. `[R,sigma] = corrcov(C)` also returns a column vector of standard deviations.",
    keywords: &["corrcov", "covariance", "correlation", "standard deviation", "statistics"],
    related: &["cov", "corrcoef", "cov2corr"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`C` must be a finite, symmetric, positive-semidefinite square covariance matrix. Diagonal entries are nonnegative variances; off-diagonal magnitudes cannot exceed the corresponding variance bound.",
                "A scalar real covariance input is treated as a 1-by-1 matrix. Symmetry and covariance bounds use floating-point tolerance.",
                "`R(i,j)` is `C(i,j)/(sigma(i)*sigma(j))`. A zero-variance denominator yields `NaN`.",
                "`sigma` is an n-by-1 column vector; unlike `cov2corr`, the correlation matrix is the first output.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Classes and execution",
            paragraphs: &[
                "Real single covariance input produces single outputs; double input produces double outputs. Real typed-integer and logical covariance inputs are RunMat-only extensions and produce double outputs.",
                "Resident real GPU input uses the provider covariance-to-correlation hook when available, preserving output ownership and residency. Supported host materialization remains available when that hook is unsupported; this is a fusion boundary.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1",
            title: "Correlation and standard deviations",
            program: "C = [4 2; 2 9]; [R,sigma] = corrcov(C)",
            display_output: Some("R = [1 0.3333; 0.3333 1]\nsigma = [2; 3]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(max(abs(R(:)-[1;1/3;1/3;1]))<1e-12); assert(isequal(size(sigma),[2 1])); assert(isequal(sigma,[2;3]));" },
        },
        BuiltinExample {
            id: "example-2",
            title: "Single-precision covariance",
            program: "C = single([1 0.5; 0.5 1]); [R,sigma] = corrcov(C)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(R),'single')); assert(strcmp(class(sigma),'single')); assert(max(abs(double(R(:))-[1;0.5;0.5;1]))<1e-6);" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What does the second output contain?", answer: "`sigma` contains `sqrt(diag(C))` as a column vector. `corrcov` returns the correlation matrix first." },
        BuiltinDocumentationFaq { question: "How are constant variables handled?", answer: "A zero diagonal variance gives a zero denominator; correlations involving that variable are `NaN`." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "cov", target: BuiltinDocumentationLinkTarget::Builtin("cov") },
        BuiltinDocumentationLink { label: "corrcoef", target: BuiltinDocumentationLinkTarget::Builtin("corrcoef") },
        BuiltinDocumentationLink { label: "cov2corr", target: BuiltinDocumentationLinkTarget::Builtin("cov2corr") },
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
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Covariance conversion semantics", location: "crates/runmat-runtime/src/builtins/stats/summary/covariance_conversion/tests.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Resident conversion ownership and errors", location: "crates/runmat-runtime/src/builtins/stats/summary/covariance_conversion/provider/tests.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled covariance conversion", location: "crates/runmat-vm/tests/stats_ml.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
