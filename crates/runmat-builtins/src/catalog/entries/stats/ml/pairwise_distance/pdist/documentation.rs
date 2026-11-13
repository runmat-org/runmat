use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("pdist"),
    slug: Some("pdist"),
    summary: "Compute pairwise distances between rows of an observation matrix.",
    description: "`pdist(X)` returns a condensed row vector containing distances between every pair of rows in `X`, ordered like the lower-left distance triangle: `(2,1), (3,1), ..., (n,1), (3,2), ...`.",
    keywords: &["pdist", "pairwise", "distance", "statistics", "machine learning", "clustering"],
    related: &["pdist2", "squareform"],
    sections: &[
        BuiltinDocumentationSection { heading: "Observations and metrics", paragraphs: &[
            "`X` must be a real numeric vector or two-dimensional matrix. Rows are observations and columns are variables. The default metric is Euclidean distance.",
            "Supported named metrics are `euclidean`, `squaredeuclidean`, `cityblock`, `chebychev`, `minkowski`, `seuclidean`, `mahalanobis`, `cosine`, `correlation`, `hamming`, `jaccard`, and `spearman`. Custom distance functions are not supported.",
            "`pdist(X,\"minkowski\",P)` uses a positive finite exponent. `pdist(X,\"seuclidean\",S)` uses a positive finite standard-deviation vector; without `S`, RunMat estimates column variances from `X`, omitting NaNs from scale estimation. `pdist(X,\"mahalanobis\",C)` accepts a covariance matrix; without `C`, RunMat computes and inverts covariance from `X`.",
        ]},
        BuiltinDocumentationSection { heading: "Classes and execution", paragraphs: &[
            "Results are double row vectors. Typed-integer observation data, distance parameters, and `CacheSize` are independently gated RunMat extensions. They must be exactly representable at the binary64 distance boundary. `CacheSize` is validated for compatibility but does not change the present CPU algorithm.",
            "Resident inputs are gathered through their owning provider and pairwise distances run on CPU. This is a materializing fusion boundary, not a GPU distance kernel.",
        ]},
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Compute Euclidean distances", program: "X = [0 0; 3 4; 4 0; 0 2]; D = pdist(X)", display_output: Some("D starts with [5 4 2], the distances from rows 2, 3, and 4 to row 1."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(D),[1 6])); assert(isequal(D(1:3),[5 4 2]));" } },
        BuiltinExample { id: "example-2", title: "Use a cityblock metric", program: "X = [0 0; 3 4; 4 0; 0 2]; D = pdist(X,\"cityblock\")", display_output: Some("D contains Manhattan distances between row pairs."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(D,[7 4 2 5 5 6]));" } },
        BuiltinExample { id: "example-3", title: "Use Minkowski distance", program: "X = [0 0; 3 4; 4 0]; D = pdist(X,\"minkowski\",3)", display_output: Some("D contains pairwise L3 distances."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(D(1)-91^(1/3))<1e-10);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Why is the output a row vector?", answer: "Condensed form avoids duplicating a symmetric matrix. Use `squareform` to expand it." },
        BuiltinDocumentationFaq { question: "Are custom distance functions supported?", answer: "No. RunMat currently supports the named numeric distance metrics listed above." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "pdist2", target: BuiltinDocumentationLinkTarget::Builtin("pdist2") },
        BuiltinDocumentationLink { label: "squareform", target: BuiltinDocumentationLinkTarget::Builtin("squareform") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/distance.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Pairwise distance metric behavior", location: "builtins::stats::ml::distance::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Integer and numeric distance integration", location: "crates/runmat-vm/tests/integer_numeric_transform_semantics.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
