use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("stats"),
    slug: Some("stats"),
    summary: "Return cache entries and hit/miss counters for a MemoizedFunction object.",
    description: "`S = stats(memoizedFcn)` inspects a `MemoizedFunction` object and returns its cached input/output entries plus hit and miss counters. The argument itself must be a memoized-function object, so integers are not a direct input class. Integer values stored inside cached arguments or results remain opaque payloads and are returned without numeric conversion.",
    keywords: &["stats",
        "memoize",
        "MemoizedFunction",
        "cache statistics"],
    related: &["memoize",
        "clearCache",
        "clearAllMemoizedCaches"],
    sections: &[BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &["`S.Cache.Inputs` is a cell row vector whose entries are cell arrays of cached input arguments.",
                "`S.Cache.Nargout` is a numeric row vector containing the requested output count for each cache entry.",
                "`S.Cache.Outputs` is a cell row vector whose entries are cell arrays of cached outputs.",
                "`S.Cache.HitCount` is a numeric row vector with per-entry hit counts.",
                "`S.Cache.TotalHits` and `S.Cache.TotalMisses` report aggregate counters.",
                "If `CacheSize` has been reduced, stats trims old entries before reporting the cache.",
                "Cached integer arguments and outputs preserve their native class and exact payload. Counter fields describe cache metadata and are not derived from cached numeric storage."]
        },
        BuiltinDocumentationSection {
            heading: "Execution and placement",
            paragraphs: &["Host-side cache inspection only."]
        },
        BuiltinDocumentationSection {
            heading: "Execution limits",
            paragraphs: &["Cache bookkeeping uses the current runtime thread. Constructing or inspecting a wrapper does not invoke the wrapped function. Calling the wrapper uses normal callable execution, including that function’s effects, requested outputs, residency, errors and cancellation.",
                "Cache and snapshot allocation is not yet fully owned for finite Managed execution; this API is not evidence of finite cache-memory support."]
        },
        BuiltinDocumentationSection {
            heading: "Supported statistics",
            paragraphs: &["RunMat reports the Cache fields listed above, including its requested-output-count key. MATLAB’s MostHitCachedInput, CacheHitRatePercent and CacheOccupancyPercent fields are not currently provided."]
        }
    ],
    examples: &[BuiltinExample {
            id: "example-1",
            title: "Inspect memoized-function cache state",
            program: "clearAllMemoizedCaches;\nf = memoize(@sqrt);\nf(9);\nf(9);\nS = stats(f)",
            display_output: Some("% S.Cache.TotalHits is 1 and S.Cache.TotalMisses is 1."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(S.Cache.TotalHits==1 && S.Cache.TotalMisses==1 && numel(S.Cache.Inputs)==1 && isequal(S.Cache.Nargout,0));"
            }
        }
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
            question: "Does `stats` change cache counters?",
            answer: "No. It only reports existing cache state, except that it enforces the current `CacheSize` by trimming old entries if necessary."
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
            label: "clearAllMemoizedCaches",
            target: BuiltinDocumentationLinkTarget::Documentation("./clearAllMemoizedCaches")
        },
        BuiltinDocumentationLink {
            label: "MATLAB reference",
            target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.lang.memoizedfunction.stats.html")
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
