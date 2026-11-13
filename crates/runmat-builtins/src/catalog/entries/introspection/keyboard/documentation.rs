use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("keyboard"),
    slug: Some("keyboard"),
    summary: "Compatibility no-op for MATLAB keyboard debugging stops.",
    description: "`keyboard` is accepted so MATLAB scripts containing manual debug stops continue to run under RunMat.",
    keywords: &["keyboard",
        "debug",
        "breakpoint"],
    related: &["dbstack"],
    sections: &[BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &["`keyboard` accepts no inputs and returns no outputs.",
                "RunMat currently treats it as a deterministic no-op. It does not enter MATLAB's interactive `K>>` debug prompt.",
                "Interactive debugger suspension is a broader runtime/REPL feature and is not implemented by this builtin."]
        },
        BuiltinDocumentationSection {
            heading: "Execution and placement",
            paragraphs: &["Host-side debugger control only.",
                "`keyboard` is control-flow debugger compatibility and has no GPU/provider behavior."]
        }
    ],
    examples: &[BuiltinExample {
            id: "example-1",
            title: "Accept a manual debug stop without halting batch execution",
            program: "keyboard",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "after=7; assert(after==7);"
            }
        }
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
            question: "Does RunMat enter an interactive debugger for `keyboard`?",
            answer: "No. The command is accepted as a no-op until interactive debugger infrastructure exists."
        }
    ],
    links: &[BuiltinDocumentationLink {
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
