use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("clearCache"),
    slug: Some("clearcache"),
    summary: "Clear the cache and counters for one MemoizedFunction object.",
    description: "`clearCache(memoizedFcn)` removes all cached entries for a `MemoizedFunction` object and resets `TotalHits` and `TotalMisses` to zero.",
    keywords: &["clearCache",
        "memoize",
        "MemoizedFunction",
        "cache"],
    related: &["memoize",
        "clearAllMemoizedCaches",
        "stats"],
    sections: &[BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &["The input must be a `MemoizedFunction` object returned by `memoize`.",
                "Cache state is shared per wrapped function, so clearing one object also clears other memoized handles for the same function.",
                "The operation exposes no output argument; requesting one is an error.",
                "Object properties such as `Function`, `Enabled`, and `CacheSize` are preserved."]
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
            title: "Clear one memoized cache",
            program: "clearAllMemoizedCaches;\nf = memoize(@sqrt);\nf(4);\nf(4);\nclearCache(f);\nS = stats(f)",
            display_output: Some("% S.Cache.TotalHits and TotalMisses are zero."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(S.Cache.TotalHits==0 && S.Cache.TotalMisses==0 && isempty(S.Cache.Inputs));"
            }
        }
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
            question: "Does `clearCache` delete the MemoizedFunction object?",
            answer: "No. It only clears cache entries and counters."
        },
        BuiltinDocumentationFaq {
            question: "Does it affect other functions?",
            answer: "No. Use `clearAllMemoizedCaches` to clear every memoized-function cache."
        }
    ],
    links: &[BuiltinDocumentationLink {
            label: "memoize",
            target: BuiltinDocumentationLinkTarget::Documentation("./memoize")
        },
        BuiltinDocumentationLink {
            label: "clearAllMemoizedCaches",
            target: BuiltinDocumentationLinkTarget::Documentation("./clearAllMemoizedCaches")
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
