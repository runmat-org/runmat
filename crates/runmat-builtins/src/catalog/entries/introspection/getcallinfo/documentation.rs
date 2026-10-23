use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("getcallinfo"),
    slug: Some("getcallinfo"),
    summary: "Return deterministic information about the current RunMat call context.",
    description: "`getcallinfo` is an undocumented MATLAB-adjacent compatibility helper. RunMat exposes deterministic call-context metadata for client code that probes it.",
    keywords: &["getcallinfo",
        "debug",
        "call",
        "stack"],
    related: &["dbstack"],
    sections: &[BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &["`getcallinfo` accepts no inputs.",
                "The result is a scalar struct with `name`, `file`, `line`, and `stack` fields.",
                "`stack` is the same N-by-1 structure array of frame records returned by `dbstack`."]
        },
        BuiltinDocumentationSection {
            heading: "Execution and placement",
            paragraphs: &["Host-side debugger introspection only.",
                "`getcallinfo` is host-side call-context introspection and has no GPU/provider behavior."]
        }
    ],
    examples: &[BuiltinExample {
            id: "example-1",
            title: "Read the current call context",
            program: "info = getcallinfo;",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isstruct(info) && all(isfield(info,{'name','file','line','stack'})) && isstruct(info.stack));"
            }
        }
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
            question: "Is `getcallinfo` full MATLAB API compatibility?",
            answer: "No public MathWorks API contract was used for this helper. RunMat provides stable current-frame and stack metadata for compatibility with code that checks for its existence."
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
