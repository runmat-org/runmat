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
    title: Some("predecessors"),
    slug: Some("predecessors"),
    summary: "Return predecessor nodes.",
    description: "`predecessors(G,node)` returns incoming neighbors for digraphs and adjacent neighbors for undirected graphs.",
    keywords: &[
        "predecessors",
        "digraph",
        "graph",
        "nodes",
    ],
    related: &[
        "graph",
        "digraph",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Graph values and results",
            paragraphs: &[
                "`predecessors(G,node)` returns incoming neighbors for digraphs and adjacent neighbors for undirected graphs.",
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
            title: "Return predecessor nodes.",
            program: "G = digraph([1 2], [3 3]); P = predecessors(G, 3);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(P, [1;2]));",
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
            label: "digraph",
            target: BuiltinDocumentationLinkTarget::Builtin("digraph"),
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
