use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("munlock"),
    slug: Some("munlock"),
    summary: "Remove a function or script from RunMat's lock compatibility registry.",
    description: "`munlock` clears lock state recorded by `mlock`.",
    keywords: &["munlock",
        "persistent",
        "lock",
        "function"],
    related: &["mlock",
        "mislocked"],
    sections: &[BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &["`munlock` with no input unlocks the active function or script.",
                "`munlock(fun)` unlocks the named function or script.",
                "Unlocking an unlocked name is allowed and has no effect."]
        },
        BuiltinDocumentationSection {
            heading: "Execution and placement",
            paragraphs: &["Host-side function-state compatibility only.",
                "`munlock` is host-side execution-state bookkeeping and has no GPU/provider behavior."]
        }
    ],
    examples: &[BuiltinExample {
            id: "example-1",
            title: "Unlock a named function",
            program: "munlock('myFunction')",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(~mislocked('myFunction'));"
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
            label: "mislocked",
            target: BuiltinDocumentationLinkTarget::Documentation("./mislocked")
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
