use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("dbclear"),
    slug: Some("dbclear"),
    summary: "Clear RunMat debugger breakpoint status entries.",
    description: "`dbclear` accepts MATLAB-style breakpoint-clear commands.",
    keywords: &["dbclear",
        "debug",
        "breakpoints"],
    related: &["dbstatus"],
    sections: &[BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &["`dbclear all`, `dbclear in file`, `dbclear in file at location`, and `dbclear if condition` command text is accepted.",
                "RunMat does not yet implement execution breakpoints or `dbstop`, so clearing the current empty registry is a no-op.",
                "`dbclear` is a sink and returns no output."]
        },
        BuiltinDocumentationSection {
            heading: "Execution and placement",
            paragraphs: &["Host-side debugger state only.",
                "`dbclear` is host-side debugger state and has no GPU/provider behavior."]
        },
        BuiltinDocumentationSection {
            heading: "Compatibility command text",
            paragraphs: &["Text arguments are admitted as compatibility words, not interpreted into executable breakpoint records. Unknown text does not create a breakpoint, condition or debugger action."]
        }
    ],
    examples: &[BuiltinExample {
            id: "example-1",
            title: "Clear all breakpoint state",
            program: "dbclear all",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "b=dbstatus; assert(iscell(b) && isempty(b));"
            }
        }
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
            question: "Does `dbclear` remove active execution breakpoints?",
            answer: "RunMat does not yet have active execution breakpoints. The command is accepted and clears the currently empty registry."
        }
    ],
    links: &[BuiltinDocumentationLink {
            label: "dbstatus",
            target: BuiltinDocumentationLinkTarget::Documentation("./dbstatus")
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
