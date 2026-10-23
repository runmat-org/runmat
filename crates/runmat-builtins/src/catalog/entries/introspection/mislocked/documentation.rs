use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("mislocked"),
    slug: Some("mislocked"),
    summary: "Return whether a function or script is in RunMat's lock compatibility registry.",
    description: "`mislocked` queries the lock state recorded by `mlock`.",
    keywords: &["mislocked",
        "mlock",
        "function",
        "lock"],
    related: &["mlock",
        "munlock"],
    sections: &[BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &["`mislocked` with no input checks the active function or script.",
                "`mislocked(fun)` checks the named function or script.",
                "The result is a logical scalar."]
        },
        BuiltinDocumentationSection {
            heading: "Execution and placement",
            paragraphs: &["Host-side function-state compatibility only.",
                "`mislocked` is host-side execution-state bookkeeping and has no GPU/provider behavior."]
        }
    ],
    examples: &[BuiltinExample {
            id: "example-1",
            title: "Check a named function",
            program: "tf = mislocked('myFunction')",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(islogical(tf) && isscalar(tf));"
            }
        }
    ],
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink {
            label: "mlock",
            target: BuiltinDocumentationLinkTarget::Documentation("./mlock")
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
