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
    title: Some("distances"),
    slug: Some("distances"),
    summary: "Compute shortest-path distance matrices.",
    description: "`distances(G)` computes all-pairs shortest-path distances. Optional source and target node lists restrict the rows and columns.",
    keywords: &[
        "distances",
        "graph",
        "dijkstra",
    ],
    related: &[
        "graph",
        "digraph",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Graph values and results",
            paragraphs: &[
                "`distances(G)` computes all-pairs shortest-path distances. Optional source and target node lists restrict the rows and columns.",
                "Source and target selectors may be numeric node vectors or name lists. Unreachable pairs have infinite distance. The result is a double matrix with one row per selected source and one column per selected target.",
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
            title: "Compute shortest-path distance matrices.",
            program: "G = graph([1 2], [2 3]); D = distances(G);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(D), [3 3])); assert(D(1,3) == 2);",
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
