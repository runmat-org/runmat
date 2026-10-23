use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("memoize"),
    slug: Some("memoize"),
    summary: "Create a MemoizedFunction handle object that caches function-handle results by input values and requested output count.",
    description: "`memoizedFcn = memoize(fh)` returns a `MemoizedFunction` handle object for function handle `fh`. Calling the object reuses cached outputs when the input values and requested output count match a previous call. Repeated `memoize` calls for the same function handle share the same object state within the current runtime thread.",
    keywords: &["memoize",
        "MemoizedFunction",
        "function handle",
        "cache",
        "performance"],
    related: &["clearCache",
        "clearAllMemoizedCaches",
        "stats",
        "feval"],
    sections: &[BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &["`fh` must be a function handle or anonymous-function closure.",
                "The `memoize` constructor itself has no integer argument or result role and is explicitly integer-inapplicable. Integer values supplied later to the returned callable remain exact cache-key and cached-result payloads.",
                "The returned object is callable like the original function through normal function-call syntax or `feval`.",
                "Cache entries are keyed by the input values and the number of requested outputs.",
                "Numeric `NaN` values compare equal for cache lookup.",
                "`Function` stores the wrapped function handle and is read-only.",
                "`Enabled` defaults to true. When false, calls dispatch to the wrapped function without reading, writing, or updating cache statistics.",
                "`CacheSize` defaults to 10 and must be a positive integer scalar. The oldest entries are evicted when the cache exceeds this size.",
                "`clearCache(memoizedFcn)` clears one shared cache and resets its counters.",
                "`clearAllMemoizedCaches` clears every memoized-function cache in the current runtime thread.",
                "`stats(memoizedFcn)` returns a structure with `Cache.Inputs`, `Cache.Nargout`, `Cache.Outputs`, `Cache.HitCount`, `Cache.TotalHits`, and `Cache.TotalMisses`."]
        },
        BuiltinDocumentationSection {
            heading: "Execution and placement",
            paragraphs: &["Host-side function-dispatch helper. GPU behavior belongs to the wrapped function body."]
        },
        BuiltinDocumentationSection {
            heading: "Callable state",
            paragraphs: &["`fh` is a function handle or anonymous function.",
                "`memoizedFcn.Function`, `memoizedFcn.Enabled`, and `memoizedFcn.CacheSize` expose the memoized-function state."]
        },
        BuiltinDocumentationSection {
            heading: "Execution limits",
            paragraphs: &["Cache bookkeeping uses the current runtime thread. Constructing or inspecting a wrapper does not invoke the wrapped function. Calling the wrapper uses normal callable execution, including that function’s effects, requested outputs, residency, errors and cancellation.",
                "Cache and snapshot allocation is not yet fully owned for finite Managed execution; this API is not evidence of finite cache-memory support."]
        },
        BuiltinDocumentationSection {
            heading: "Syntax",
            paragraphs: &["memoizedFcn = memoize(fh)\ny = memoizedFcn(x)\n[y1, y2] = memoizedFcn(x)"]
        }
    ],
    examples: &[BuiltinExample {
            id: "example-1",
            title: "Memoize an expensive function handle",
            program: "clearAllMemoizedCaches;\nf = memoize(@(x) sum(sin(x)));\ny1 = f(1:1000);\ny2 = f(1:1000);",
            display_output: Some("% The second call reuses the cached result."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "S=stats(f); assert(isequaln(y1,y2) && S.Cache.TotalHits==1 && S.Cache.TotalMisses==1);"
            }
        },
        BuiltinExample {
            id: "example-2",
            title: "Inspect cache statistics",
            program: "clearAllMemoizedCaches;\nf = memoize(@sqrt);\nf(4);\nf(4);\nS = stats(f)",
            display_output: Some("% S.Cache.TotalHits is 1 and TotalMisses is 1."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(S.Cache.TotalHits==1 && S.Cache.TotalMisses==1);"
            }
        },
        BuiltinExample {
            id: "example-3",
            title: "Limit cache capacity",
            program: "clearAllMemoizedCaches;\nf = memoize(@cos);\nf.CacheSize = 2;\nf(1); f(2); f(3);",
            display_output: Some("% The oldest cached entry is evicted."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "S=stats(f); assert(numel(S.Cache.Inputs)==2 && S.Cache.TotalMisses==3);"
            }
        },
        BuiltinExample {
            id: "example-4",
            title: "Disable cache lookup temporarily",
            program: "clearAllMemoizedCaches;\nf = memoize(@rand);\nf.Enabled = false;\na = f(1);\nb = f(1);",
            display_output: Some("% The two calls dispatch normally and do not update cache statistics."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "S=stats(f); assert(S.Cache.TotalHits==0 && S.Cache.TotalMisses==0 && isempty(S.Cache.Inputs) && isa(a,'double') && isa(b,'double'));"
            }
        }
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
            question: "Does `memoize(@f)` return a function handle?",
            answer: "No. It returns a `MemoizedFunction` handle object that can be invoked like the original function."
        },
        BuiltinDocumentationFaq {
            question: "Do separate `memoize(@f)` calls share cache data?",
            answer: "Yes, within the current runtime thread. The cache is associated with the wrapped function identity, so repeated memoization of the same function shares cache entries and properties."
        },
        BuiltinDocumentationFaq {
            question: "Are multiple-output calls cached separately?",
            answer: "Yes. The requested output count is part of the cache key."
        },
        BuiltinDocumentationFaq {
            question: "Does this launch GPU kernels?",
            answer: "No. `memoize` itself is host-side dispatch and bookkeeping. The wrapped function may still call GPU-aware operations."
        }
    ],
    links: &[BuiltinDocumentationLink {
            label: "clearCache",
            target: BuiltinDocumentationLinkTarget::Documentation("./clearCache")
        },
        BuiltinDocumentationLink {
            label: "clearAllMemoizedCaches",
            target: BuiltinDocumentationLinkTarget::Documentation("./clearAllMemoizedCaches")
        },
        BuiltinDocumentationLink {
            label: "stats",
            target: BuiltinDocumentationLinkTarget::Documentation("./stats")
        },
        BuiltinDocumentationLink {
            label: "feval",
            target: BuiltinDocumentationLinkTarget::Documentation("./feval")
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
