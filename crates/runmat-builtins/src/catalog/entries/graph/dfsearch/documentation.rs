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
    title: Some("dfsearch"),
    slug: Some("dfsearch"),
    summary: "Return depth-first search order.",
    description: "`dfsearch(G,start)` returns the reachable nodes in depth-first order. Directed graphs follow outgoing edges.",
    keywords: &[
        "dfsearch",
        "graph",
        "depth first search",
    ],
    related: &[
        "graph",
        "digraph",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Graph values and results",
            paragraphs: &[
                "`dfsearch(G,start)` returns the reachable nodes in depth-first order. Directed graphs follow outgoing edges.",
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
            title: "Return depth-first search order.",
            program: "G = graph([1 2], [2 3]); order = dfsearch(G, 1);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(order, [1;2;3]));",
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
