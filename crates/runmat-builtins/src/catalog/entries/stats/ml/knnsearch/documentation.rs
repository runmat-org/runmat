use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("knnsearch"),
    slug: Some("knnsearch"),
    summary: "Find k-nearest neighbors in an observation matrix.",
    description: "`knnsearch(X,Y)` returns one-based row indices in `X` for the nearest neighbors of each row in `Y`.",
    keywords: &["knnsearch", "nearest neighbor", "k nearest", "statistics", "machine learning", "distance"],
    related: &["pdist2", "pdist"],
    sections: &[
        BuiltinDocumentationSection { heading: "Inputs and results", paragraphs: &[
            "`X` and `Y` may be real numeric or logical vectors or two-dimensional matrices with the same number of columns. Logical data is converted to numeric distances. The default search is exhaustive with `K=1` and Euclidean distance.",
            "`[Idx,D] = knnsearch(...)` returns indices first and distances second. Without `IncludeTies`, both outputs are `size(Y,1)` by `K` double matrices.",
            "Named distance metrics are `euclidean`, `squaredeuclidean`, `cityblock`, `chebychev`, `minkowski`, `seuclidean`, `mahalanobis`, `cosine`, `correlation`, `hamming`, `jaccard`, and `spearman`. The fast Euclidean aliases are accepted and evaluated with the same exact Euclidean implementations. Custom distance functions are not supported.",
            "`P`, `Scale`, and `Cov` provide parameters for Minkowski, standardized Euclidean, and Mahalanobis distance respectively.",
            "`IncludeTies`,true returns `size(Y,1)` by 1 cell arrays; each element contains all neighbors tied at the K-th distance. Tied outputs remain sorted by distance even when `SortIndices`,false is supplied.",
        ]},
        BuiltinDocumentationSection { heading: "Options, classes, and execution", paragraphs: &[
            "`NSMethod`, `BucketSize`, `CacheSize`, and `SortIndices` are parsed for compatibility. `NSMethod`,`kdtree` is admitted only for Euclidean, cityblock, Chebychev, and Minkowski distances. RunMat currently evaluates nearest neighbors exhaustively and returns sorted neighbors; it does not build a kd-tree.",
            "Typed-integer X, Y, K, BucketSize, metric parameters, and logical controls are independently gated RunMat extensions. Native integer structural counts decode exactly; legacy floating counts within 1e-12 above an integer are accepted and truncated. Numerical distance roles must be exactly representable at the binary64 boundary.",
            "Supported floating resident input gathers through its owning provider for CPU nearest-neighbor search, then restores numeric or tied-cell outputs to that provider. Explicit residency must not silently become host residency. There is no nearest-neighbor GPU kernel or fused path.",
            "For MATLAB-compatible explicit `gpuArray` calls, use `NSMethod`,`exhaustive`, default `IncludeTies` and `SortIndices`, and no fast-distance alias. Other explicit GPU forms require RunMat mode. Automatic residency remains transparent.",
        ]},
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Find nearest rows", program: "X = [0 0; 2 0; 5 0]; Y = [1 0; 4 0]; Idx = knnsearch(X,Y)", display_output: Some("Idx is a 2-by-1 matrix of row indices from X."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(Idx,[1;3]));" } },
        BuiltinExample { id: "example-2", title: "Request distances and two neighbors", program: "X = [0 0; 2 0; 5 0]; Y = [1 0; 4 0]; [Idx,D] = knnsearch(X,Y,\"K\",2)", display_output: Some("Idx and D are size(Y,1)-by-2 matrices."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(Idx),[2 2])); assert(isequal(size(D),[2 2])); assert(isequal(Idx(:,1),[1;3]));" } },
        BuiltinExample { id: "example-3", title: "Include tied neighbors", program: "Idx = knnsearch([0;1;-1],0,\"K\",2,\"IncludeTies\",true)", display_output: Some("Idx is a cell array containing the selected neighbor indices."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(iscell(Idx)); assert(isequal(Idx{1},[1 2 3]));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does RunMat build a kd-tree?", answer: "No. It parses kd-tree-related options for compatibility but currently uses exhaustive search." },
        BuiltinDocumentationFaq { question: "Are custom distance functions supported?", answer: "No. RunMat currently supports named numeric distance metrics." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "pdist2", target: BuiltinDocumentationLinkTarget::Builtin("pdist2") },
        BuiltinDocumentationLink { label: "pdist", target: BuiltinDocumentationLinkTarget::Builtin("pdist") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/distance.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Neighbor selection and output ownership", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/distance/selection.rs") },
            BuiltinDocumentationLink { label: "Distance options", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/distance/options.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "K-nearest selection, options, and residency", location: "builtins::stats::ml::distance::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "VM statistics integration", location: "crates/runmat-vm/tests/stats_ml.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
