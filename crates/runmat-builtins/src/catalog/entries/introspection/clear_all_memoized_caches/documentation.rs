use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("clearAllMemoizedCaches"),
    slug: Some("clearallmemoizedcaches"),
    summary: "Clear all MemoizedFunction caches in the current runtime thread.",
    description: "`clearAllMemoizedCaches` removes cached entries and resets counters for every `MemoizedFunction` registered in the current runtime thread.",
    keywords: &["clearAllMemoizedCaches",
        "memoize",
        "MemoizedFunction",
        "cache"],
    related: &["memoize",
        "clearCache",
        "stats"],
    sections: &[BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &["Clears all memoized-function caches known to the runtime.",
                "Preserves the `MemoizedFunction` objects and their properties.",
                "The operation returns zero internally and suppresses automatic output in statement form."]
        },
        BuiltinDocumentationSection {
            heading: "Execution and placement",
            paragraphs: &["Host-side cache bookkeeping only."]
        },
        BuiltinDocumentationSection {
            heading: "Execution limits",
            paragraphs: &["Cache bookkeeping uses the current runtime thread. Constructing or inspecting a wrapper does not invoke the wrapped function. Calling the wrapper uses normal callable execution, including that function’s effects, requested outputs, residency, errors and cancellation.",
                "Cache and snapshot allocation is not yet fully owned for finite Managed execution; this API is not evidence of finite cache-memory support."]
        }
    ],
    examples: &[BuiltinExample {
            id: "example-1",
            title: "Clear every memoized cache",
            program: "clearAllMemoizedCaches;\nf = memoize(@sin);\ng = memoize(@cos);\nf(1); g(1);\nclearAllMemoizedCaches",
            display_output: Some("% Both caches are empty."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "S=stats(f); T=stats(g); assert(isempty(S.Cache.Inputs) && isempty(T.Cache.Inputs) && S.Cache.TotalHits==0 && T.Cache.TotalMisses==0);"
            }
        }
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
            question: "Does this remove MemoizedFunction objects?",
            answer: "No. Existing objects remain valid; only their cache entries and counters are reset."
        }
    ],
    links: &[BuiltinDocumentationLink {
            label: "memoize",
            target: BuiltinDocumentationLinkTarget::Documentation("./memoize")
        },
        BuiltinDocumentationLink {
            label: "clearCache",
            target: BuiltinDocumentationLinkTarget::Documentation("./clearCache")
        },
        BuiltinDocumentationLink {
            label: "stats",
            target: BuiltinDocumentationLinkTarget::Documentation("./stats")
        }
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
                label: "`crates/runmat-runtime/src/builtins/introspection/memoize.rs`",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/memoize.rs")
            }
        ],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Runtime semantic tests",
            location: "crates/runmat-runtime/src/builtins/introspection/memoize.rs::tests",
        }],
        notes: &[]
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial)
}
;
