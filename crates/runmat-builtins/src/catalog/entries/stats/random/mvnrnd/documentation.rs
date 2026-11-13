//! Conserved reference prose and independently asserted examples.
use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`mu` can be a scalar, a vector, or an m-by-d matrix with one mean vector per row. A d-by-1 column vector is treated as d one-dimensional means.",
            "`Sigma` can be a d-by-d covariance matrix, a 1-by-d diagonal-variance vector, a d-by-d-by-m array of covariance pages, or a 1-by-d-by-m array of diagonal-variance pages.",
            "`mvnrnd(mu,Sigma,n)` draws `n` rows when `mu` is a single mean vector and `Sigma` has one covariance page; `n` must be a positive scalar integer.",
            "Current MATLAB documents single and double mu, Sigma, and n values. Typed integers and logical inputs are independently gated RunMat extensions; integer distribution values must be exactly representable as double.",
            "A single mu or Sigma selects single output; otherwise output is double. Integer n is structural and does not select output class or residency.",
            "Without `n`, RunMat draws one row for each row of `mu`, each page of `Sigma`, or their shared row/page count.",
            "A single covariance matrix is reused for all rows of matrix `mu`; a single mean vector is reused for all covariance pages.",
            "`Sigma` must be symmetric positive semidefinite. Positive-definite covariances use Cholesky; singular semidefinite covariances use a symmetric eigen factor.",
            "All distribution inputs must be real finite numeric values, and outputs place observations in rows.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution and residency",
        paragraphs: &["Sampling currently uses the host RNG and covariance-factor path. Resident mu or Sigma gathers without consuming the source and restores through the exact owner only when it can preserve the required class; incompatible automatic residency may finish on host and incompatible explicit gpuArray intent errors."],
    },
    BuiltinDocumentationSection {
        heading: "Random state and fusion",
        paragraphs: &["Random generation and RNG state control are stateful boundaries and are not fused into elementwise kernels."],
    },
    BuiltinDocumentationSection {
        heading: "Storage precision",
        paragraphs: &["Supported sample/storage class declaration: f32. See the distribution, integer-extension, and residency rules above."],
    },
    BuiltinDocumentationSection {
        heading: "Storage precision",
        paragraphs: &["Supported sample/storage class declaration: f64. See the distribution, integer-extension, and residency rules above."],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Draw five bivariate samples",
        program: "rng('default');\nR = mvnrnd([1 2], [4 1; 1 9], 5)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(R,'double')); assert(isequal(size(R),[5 2])); assert(all(isfinite(R(:))));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Use one covariance page per mean row",
        program: "rng(0);\nmu = [0 0; 10 20];\nSigma = cat(3, eye(2), [4 0; 0 9]);\nR = mvnrnd(mu, Sigma)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(R),[2 2])); assert(all(isfinite(R(:))));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Use diagonal variances",
        program: "rng(0);\nR = mvnrnd([1 2], [4 9], 5)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(R),[5 2])); assert(all(isfinite(R(:))));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Draw from a singular covariance",
        program: "rng(0);\nR = mvnrnd([0 0], [1 1; 1 1], 3)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(R),[3 2])); assert(max(abs(R(:,1)-R(:,2))) < 1e-10);",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Retain single precision with a singular zero covariance",
        program: "rng(0); R = mvnrnd(single([2 3]),single(zeros(2)),3);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(R,'single')); assert(isequal(R,single([2 3;2 3;2 3])));",
        },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("mvnrnd"),
    slug: Some("mvnrnd"),
    summary: "Generate multivariate normal random samples.",
    description: "`mvnrnd(mu,Sigma)` draws rows from one or more multivariate normal distributions. `mu` supplies mean vectors and `Sigma` supplies covariance matrices.",
    keywords: &["mvnrnd", "multivariate normal", "gaussian", "random", "statistics"],
    related: &[],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does mvnrnd use the same RNG as randn and normrnd?",
            answer: "Yes. RunMat draws standard normal variates from the shared deterministic RNG state, so `rng` controls reproducibility.",
        },
        BuiltinDocumentationFaq {
            question: "Does RunMat require Sigma to be positive definite?",
            answer: "No. Positive semidefinite covariance matrices are accepted. Negative eigenvalues outside numerical tolerance raise an error.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "normrnd",
            target: BuiltinDocumentationLinkTarget::Builtin("normrnd"),
        },
        BuiltinDocumentationLink {
            label: "randn",
            target: BuiltinDocumentationLinkTarget::Builtin("randn"),
        },
        BuiltinDocumentationLink {
            label: "rng",
            target: BuiltinDocumentationLinkTarget::Builtin("rng"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/stats/random/mvnrnd.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/random/mvnrnd.rs"),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::stats::random::mvnrnd::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "integration",
                location: "crates/runmat-vm/tests/integer_normal_distribution_and_regression_semantics.rs",
            },
        ],
        notes: &["Integer compatibility and placement are declared in the canonical contract. Examples assert shape, class, bounds, or state replay rather than fixed random transcripts."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
