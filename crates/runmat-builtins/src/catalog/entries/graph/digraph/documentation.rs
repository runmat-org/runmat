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
    title: Some("digraph"),
    slug: Some("digraph"),
    summary: "Create a directed graph object.",
    description: "`digraph` creates a directed graph compatibility object with `Nodes` and `Edges` tables. It supports edge-list construction with numeric or named nodes, optional weights, optional node names, and dense adjacency matrices.",
    keywords: &[
        "digraph",
        "directed graph",
        "nodes",
        "edges",
        "network",
    ],
    related: &[
        "graph",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Graph values and results",
            paragraphs: &[
                "`digraph` creates a directed graph compatibility object with `Nodes` and `Edges` tables. It supports edge-list construction with numeric or named nodes, optional weights, optional node names, and dense adjacency matrices.",
                "Direction is preserved by `successors`, `predecessors`, `indegree`, `outdegree`, `shortestpath`, `distances`, `bfsearch`, `dfsearch`, and `toposort`.",
                "Node ids are positive one-based integers. Fixed-width integer ids are decoded exactly; edge and adjacency weights must be finite, nonnegative, and exactly representable as double before the host graph stores them. Directed adjacency entry A(i,j) defines the edge i to j.",
                "Typed-integer adjacency arrays and typed-integer edge weights require RunMat compatibility mode and exact double representation. Fixed-width node indices remain structural controls in either mode; MATLAB's public numeric-index description does not establish per-class parity.",
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
            title: "Create a directed graph object.",
            program: "G = digraph([1 2], [2 3]); n = numedges(G);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(n == 2);",
            },
        },
        BuiltinExample {
            id: "integer-adjacency",
            title: "Construct a directed graph from a typed integer adjacency matrix in RunMat mode.",
            program: "G = digraph(uint16([0 2; 0 0])); A = adjacency(G);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(A(1,2) == 2); assert(A(2,1) == 0);",
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
            label: "graph",
            target: BuiltinDocumentationLinkTarget::Builtin("graph"),
        },
        BuiltinDocumentationLink {
            label: "MATLAB digraph input classes",
            target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/digraph.html"),
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
