use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("linkage"),
    slug: Some("linkage"),
    summary: "Construct a hierarchical agglomerative clustering tree.",
    description: "`linkage(X)` clusters rows of an observation matrix; `linkage(Y,method)` clusters a condensed distance vector in `pdist` order.",
    keywords: &["linkage", "hierarchical", "cluster", "clustering", "dendrogram", "statistics", "machine learning"],
    related: &["pdist", "squareform"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Inputs and tree output",
            paragraphs: &[
                "Rows of `X` are observations and columns are variables; at least two observations are required. An n-by-1 column is n one-dimensional observations. A row vector is interpreted as condensed pairwise distances and must have length n*(n-1)/2.",
                "All distances must be finite and nonnegative. For `centroid`, `median`, and `ward`, condensed distances must also pass a Euclidean-distance validity check.",
                "`Z` has one merge per row and three columns. Columns one and two are one-based cluster identifiers; original observations are 1:m, then new clusters are m+1 onward. Column three is the merge distance.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Methods, metrics, and controls",
            paragraphs: &[
                "The default method is `single`. Supported methods are `single`, `complete`, `average`, `weighted`, `centroid`, `median`, and `ward`, including the aliases `nearest`, `farthest`, `upgma`, `wpgma`, `upgmc`, `wpgmc`, and `wardlinkage`.",
                "For observation matrices, `linkage(X,method,metric,...)` accepts the named metrics and metric parameters used by `pdist`. A cell array may carry the complete `pdist` metric arguments. Condensed input does not accept a metric.",
                "`SaveMemory`,`off` is accepted. `SaveMemory`,`on` is admitted only for an observation matrix with centroid, median, or ward linkage and Euclidean distance; both settings currently use the bounded in-memory implementation.",
                "The current exact agglomerative implementation admits at most 700 observations. This is an existing resource limitation, not a general scaling claim.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Numeric classes and execution",
            paragraphs: &[
                "Typed-integer observations, condensed distances, and numeric metric parameters are separately gated RunMat extensions and must be exactly representable at the binary64 clustering boundary. The returned tree is a double matrix.",
                "Clustering runs on CPU and returns a host tree. Automatically resident data may gather through its owner. Explicit gpuArray input requires RunMat mode and uses an owning-provider host fallback; no linkage GPU kernel or fused path is claimed.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "example-1", title: "Cluster observation rows",
            program: "X = [0 0; 3 4; 4 0; 0 2]; Z = linkage(X,\"single\")",
            display_output: Some("Z is 3-by-3; the first merge joins observations 1 and 4 at distance 2."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(Z),[3 3])); assert(isequal(Z(1,1:2),[1 4])); assert(Z(1,3)==2);" },
        },
        BuiltinExample {
            id: "example-2", title: "Cluster condensed distances",
            program: "Y = [1 4 6 5 7 2]; Z = linkage(Y,\"complete\")",
            display_output: Some("Z describes complete-linkage merges for four observations."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(Z),[3 3])); assert(Z(1,3)==1);" },
        },
        BuiltinExample {
            id: "example-3", title: "Choose a pdist metric",
            program: "X = [0 0; 3 4; 4 0; 0 2]; Z = linkage(X,\"average\",\"cityblock\")",
            display_output: Some("Pairwise cityblock distances are used before average-linkage clustering."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(Z),[3 3])); assert(Z(1,3)==2);" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does linkage draw a dendrogram?", answer: "No. It returns the tree matrix; dendrogram rendering is a separate function." },
        BuiltinDocumentationFaq { question: "Are custom distance functions supported?", answer: "No. RunMat currently supports the named numeric metrics shared with pdist." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "pdist", target: BuiltinDocumentationLinkTarget::Builtin("pdist") },
        BuiltinDocumentationLink { label: "squareform", target: BuiltinDocumentationLinkTarget::Builtin("squareform") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/linkage.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Clustering admission", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/linkage/admission.rs") },
            BuiltinDocumentationLink { label: "Hierarchical solver", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/stats/ml/linkage/compute.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Hierarchical merge and admission tests", location: "builtins::stats::ml::linkage::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "VM ML integration", location: "crates/runmat-vm/tests/stats_ml.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
