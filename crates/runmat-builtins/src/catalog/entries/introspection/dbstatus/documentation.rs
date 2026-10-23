use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("dbstatus"),
    slug: Some("dbstatus"),
    summary: "Return RunMat debugger breakpoint status.",
    description: "`dbstatus` returns RunMat's breakpoint registry state.",
    keywords: &["dbstatus",
        "debug",
        "breakpoints"],
    related: &["dbclear",
        "dbstack"],
    sections: &[BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &["`dbstatus` accepts MATLAB-style file and `'-completenames'` text arguments.",
                "RunMat does not yet implement execution breakpoints or `dbstop`, so the current status is an empty 0-by-0 cell list.",
                "Until breakpoint records are implemented, the empty result uses the existing 0-by-0 cell-list contract."]
        },
        BuiltinDocumentationSection {
            heading: "Execution and placement",
            paragraphs: &["Host-side debugger state only.",
                "`dbstatus` is host-side debugger state and has no GPU/provider behavior."]
        },
        BuiltinDocumentationSection {
            heading: "Compatibility command text",
            paragraphs: &["Text arguments are admitted as compatibility words, not interpreted into executable breakpoint records. Unknown text does not create a breakpoint, condition or debugger action."]
        }
    ],
    examples: &[BuiltinExample {
            id: "example-1",
            title: "Query breakpoint state",
            program: "b = dbstatus;",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(iscell(b) && isequal(size(b),[0 0]));"
            }
        }
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
            question: "Does `dbstatus` list MATLAB breakpoints?",
            answer: "Not yet. The interactive breakpoint engine is a broader runtime feature; this compatibility helper returns the currently empty registry."
        }
    ],
    links: &[BuiltinDocumentationLink {
            label: "dbclear",
            target: BuiltinDocumentationLinkTarget::Documentation("./dbclear")
        },
        BuiltinDocumentationLink {
            label: "dbstack",
            target: BuiltinDocumentationLinkTarget::Documentation("./dbstack")
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
