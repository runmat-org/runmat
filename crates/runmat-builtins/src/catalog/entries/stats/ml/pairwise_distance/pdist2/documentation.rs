use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("pdist2"),
    slug: Some("pdist2"),
    summary: "Compute pairwise distances between two sets of observations.",
    description: "`pdist2(X,Y)` returns a `size(X,1)` by `size(Y,1)` matrix of distances between rows of `X` and rows of `Y`.",
    keywords: &["pdist2", "pairwise", "distance", "nearest neighbor", "statistics", "machine learning"],
    related: &["pdist", "squareform"],
    sections: &[
        BuiltinDocumentationSection { heading: "Observations and metrics", paragraphs: &[
            "`X` and `Y` must be real numeric vectors or two-dimensional matrices with the same number of columns. The default metric is Euclidean distance.",
            "Supported named metrics match `pdist`: `euclidean`, `squaredeuclidean`, `cityblock`, `chebychev`, `minkowski`, `seuclidean`, `mahalanobis`, `cosine`, `correlation`, `hamming`, `jaccard`, and `spearman`. Custom distance functions are not supported.",
            "`pdist2(X,Y,\"minkowski\",P)` uses a positive finite exponent. `seuclidean` accepts a standard-deviation vector and defaults to scale estimated from `X`. `mahalanobis` accepts a symmetric positive-definite covariance matrix and defaults to covariance estimated from `X`.",
            "`\"Smallest\",K` and `\"Largest\",K` return the K smallest or largest distances for each row of `Y`; when K exceeds the number of rows in `X`, all X rows are returned. `[D,I]` also returns one-based row indices from `X` for selected distances.",
        ]},
        BuiltinDocumentationSection { heading: "Classes and execution", paragraphs: &[
            "Distances and selected indices are double. Typed-integer observation data, distance parameters, and `CacheSize` are checked RunMat floating-boundary extensions; typed `Smallest` and `Largest` counts are separately gated structural extensions decoded from native storage. `CacheSize` is validated but does not change the current CPU algorithm.",
            "Resident inputs are gathered through their owning provider, and pairwise distance kernels run on CPU. This operation is a materializing fusion boundary, not a GPU kernel.",
        ]},
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Compute distances between two sets", program: "X = [0 0; 2 0]; Y = [1 0; 3 0]; D = pdist2(X,Y)", display_output: Some("D is a 2-by-2 distance matrix."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(D,[1 3;1 1]));" } },
        BuiltinExample { id: "example-2", title: "Use squared Euclidean distance", program: "X = [0 0; 2 0]; Y = [1 0; 3 0]; D = pdist2(X,Y,\"squaredeuclidean\")", display_output: Some("D contains squared distances."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(D,[1 9;1 1]));" } },
        BuiltinExample { id: "example-3", title: "Keep nearest distances per query row", program: "X = [0 0; 2 0]; Y = [1 0; 3 0]; [D,I] = pdist2(X,Y,\"euclidean\",\"Smallest\",1)", display_output: Some("D and I are 1-by-2 rows of nearest distances and one-based indices in X."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(D,[1 1])); assert(isequal(I,[1 2]));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does Smallest return indices?", answer: "Yes. Request `[D,I]` to receive one-based row indices from `X` for the selected distances." },
        BuiltinDocumentationFaq { question: "Are custom distance functions supported?", answer: "No. RunMat currently supports the named numeric distance metrics listed above." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "pdist", target: BuiltinDocumentationLinkTarget::Builtin("pdist") },
        BuiltinDocumentationLink { label: "squareform", target: BuiltinDocumentationLinkTarget::Builtin("squareform") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/distance.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Selected distances and output indices", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/distance/selection.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Pairwise distance selection and metrics", location: "builtins::stats::ml::distance::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Integer and numeric distance integration", location: "crates/runmat-vm/tests/integer_numeric_transform_semantics.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
