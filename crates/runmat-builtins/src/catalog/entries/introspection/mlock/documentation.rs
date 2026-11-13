use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("mlock"),
    slug: Some("mlock"),
    summary: "Mark the current function or script as locked in RunMat's compatibility registry.",
    description: "`mlock` records the active function or script in RunMat's lock registry so `mislocked` and `munlock` behave predictably.",
    keywords: &["mlock",
        "persistent",
        "lock",
        "function"],
    related: &["mislocked",
        "munlock"],
    sections: &[BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &["`mlock` accepts no inputs and locks the active function or script name.",
                "The registry is thread-local execution state.",
                "RunMat does not yet evict parsed/compiled MATLAB functions in the same way MATLAB clears functions from memory, so the lock currently affects compatibility queries rather than code-cache eviction."]
        },
        BuiltinDocumentationSection {
            heading: "Execution and placement",
            paragraphs: &["Host-side function-state compatibility only.",
                "`mlock` is host-side execution-state bookkeeping and has no GPU/provider behavior."]
        }
    ],
    examples: &[BuiltinExample {
            id: "example-1",
            title: "Lock the current function",
            program: "mlock",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(mislocked); munlock; assert(~mislocked);"
            }
        }
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
            question: "Does `mlock` prevent RunMat from clearing compiled code?",
            answer: "RunMat does not currently expose MATLAB-style function-cache eviction. The lock state is implemented for `mislocked`/`munlock` compatibility."
        }
    ],
    links: &[BuiltinDocumentationLink {
            label: "mislocked",
            target: BuiltinDocumentationLinkTarget::Documentation("./mislocked")
        },
        BuiltinDocumentationLink {
            label: "munlock",
            target: BuiltinDocumentationLinkTarget::Documentation("./munlock")
        }
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
                label: "`crates/runmat-runtime/src/builtins/introspection/debugging.rs`",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/debugging.rs")
            }
        ],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Runtime semantic tests",
            location: "crates/runmat-runtime/src/builtins/introspection/debugging.rs::tests",
        }],
        notes: &[]
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial)
}
;
