use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Graph runtime implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/graph/domain/entrypoints.rs"),
};

const DOMAIN: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Graph domain and algorithms",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/graph/domain/algorithms.rs"),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("graph"),
    slug: Some("graph"),
    summary: "Create an undirected graph object.",
    description: "`graph` creates an undirected graph compatibility object with inspectable `Nodes` and `Edges` tables. It supports edge-list construction with numeric or named nodes, optional edge weights, optional node names, and dense adjacency matrices.",
    keywords: &[
        "graph",
        "nodes",
        "edges",
        "network",
    ],
    related: &[
        "digraph",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Graph values and results",
            paragraphs: &[
                "`graph` creates an undirected graph compatibility object with inspectable `Nodes` and `Edges` tables. It supports edge-list construction with numeric or named nodes, optional edge weights, optional node names, and dense adjacency matrices.",
                "Unnamed graph `Nodes` tables include an `Index` column because RunMat's current table representation cannot encode a nonzero-height table with zero variables.",
                "Edge weights must be finite nonnegative values.",
                "Node ids are positive one-based integers. Fixed-width integer ids are decoded exactly; edge and adjacency weights must be finite, nonnegative, and exactly representable as double before the host graph stores them. The optional `upper` or `lower` adjacency flag chooses one triangle; without a flag, a nonzero upper entry takes precedence and the lower entry is used when the upper entry is zero.",
                "Typed-integer adjacency arrays and typed-integer edge weights require RunMat compatibility mode. Both are checked for exact double representation before graph storage. Fixed-width node indices remain structural controls in either mode; MATLAB's public numeric-index description does not establish per-class parity.",
                "MATLAB requires a symmetric undirected adjacency input unless a triangle option is specified. RunMat's no-option upper-precedence behavior for asymmetric matrices is an existing compatibility limit.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and placement",
            paragraphs: &[
                "Graph construction, inspection, and traversal use host-side algorithms. The operation ends array fusion and does not launch a GPU graph kernel. Consumed resident numeric arguments gather through their owning provider; the result is host-resident.",
                "The operation uses the same host graph algorithms in native and browser runtimes; availability of resident-input gathering depends on the active provider.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "basic",
            title: "Create an undirected graph object.",
            program: "G = graph([1 2], [2 3]); n = numnodes(G);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(n == 3);",
            },
        },
        BuiltinExample {
            id: "integer-weights",
            title: "Use typed integer node ids and RunMat-only edge weights.",
            program: "G = graph(uint16([1 2]), uint16([2 3]), uint16([4 5])); A = adjacency(G);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(A(1,2) == 4); assert(A(2,3) == 5);",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does this run as a GPU graph kernel?",
            answer: "No. The current graph algorithms run on the host; resident inputs are materialized by the provider that owns them.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "digraph",
            target: BuiltinDocumentationLinkTarget::Builtin("digraph"),
        },
        BuiltinDocumentationLink {
            label: "MATLAB graph input classes",
            target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/graph.html"),
        },
        SOURCE,
        DOMAIN,
    ],
    media: &[
    ],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            SOURCE,
            DOMAIN,
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Graph domain and query regressions",
                location: "crates/runmat-runtime/src/builtins/graph/domain/tests.rs",
            },
        ],
        notes: &[
            "This is a host graph contract; integrated native, browser, provider, and worker validation is tracked separately.",
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
