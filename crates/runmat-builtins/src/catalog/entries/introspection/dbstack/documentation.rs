use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("dbstack"),
    slug: Some("dbstack"),
    summary: "Return or display the active RunMat call stack.",
    description: "`dbstack` inspects RunMat's runtime call-stack context and returns MATLAB-style frame metadata.",
    keywords: &["dbstack",
        "debug",
        "call stack",
        "stack trace"],
    related: &["dbstatus",
        "getcallinfo"],
    sections: &[BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &["`dbstack` with no output displays the active stack.",
                "`ST = dbstack` returns stack entries as an N-by-1 structure array.",
                "Each frame struct contains exactly the public `file`, `name`, and `line` fields; `I` is a separate second output.",
                "`dbstack(n)` omits the first `n` returned entries; both `dbstack('-completenames', n)` and `dbstack(n, '-completenames')` additionally report full source paths when the source catalog has them.",
                "`[ST,I] = dbstack(...)` returns the stack structure array and a current workspace index of `1` including when the returned stack is empty; more than two requested outputs reject."]
        },
        BuiltinDocumentationSection {
            heading: "Execution and placement",
            paragraphs: &["Host-side debugger introspection only; it does not inspect or move GPU buffers.",
                "`dbstack` is host-side introspection and is excluded from GPU execution and fusion."]
        }
    ],
    examples: &[BuiltinExample {
            id: "example-1",
            title: "Capture the current call stack",
            program: "ST = dbstack;",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isstruct(ST) && all(isfield(ST,{'file','name','line'})));"
            }
        },
        BuiltinExample {
            id: "example-2",
            title: "Skip the current frame",
            program: "ST = dbstack('-completenames', 1);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isstruct(ST));"
            }
        }
    ],
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink {
            label: "dbstatus",
            target: BuiltinDocumentationLinkTarget::Documentation("./dbstatus")
        },
        BuiltinDocumentationLink {
            label: "getcallinfo",
            target: BuiltinDocumentationLinkTarget::Documentation("./getcallinfo")
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
