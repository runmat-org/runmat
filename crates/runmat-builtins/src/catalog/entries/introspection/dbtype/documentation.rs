use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("dbtype"),
    slug: Some("dbtype"),
    summary: "Display MATLAB source text with line numbers.",
    description: "`dbtype` resolves a MATLAB source file through the active filesystem/search-path layer and writes numbered source lines to the command stream.",
    keywords: &["dbtype",
        "debug",
        "source",
        "line numbers"],
    related: &["which",
        "dbstack"],
    sections: &[BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &["`dbtype file` displays the entire source file with line numbers.",
                "`dbtype file start:end` displays an inclusive line range.",
                "The file argument may be a path, an `.m` file, or a bare function/script name resolvable on the RunMat path.",
                "Builtins cannot be displayed because there is no MATLAB source file to show."]
        },
        BuiltinDocumentationSection {
            heading: "Execution and placement",
            paragraphs: &["Host-side source-file I/O only.",
                "`dbtype` is a host-side source display command and is excluded from GPU execution and fusion."]
        }
    ],
    examples: &[BuiltinExample {
            id: "example-1",
            title: "Display a file",
            program: "dbtype myscript.m",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
                id: BuiltinExampleFixtureId {
                    local_name: "dbtype-1"
                },
                root: BuiltinFilesystemRoot::IsolatedWorkspace,
                entries: &[BuiltinFilesystemEntry::File {
                        relative_path: "myscript.m",
                        content: BuiltinFixtureContent::Utf8("a = 1;\nb = 2;\nc = 3;\nd = 4;\ne = 5;\nf = 6;\ng = 7;\nh = 8;\n")
                    }
                ]
            }
            ),
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "captured=evalc('dbtype myscript.m'); assert(contains(captured,'1') && contains(captured,'a = 1;') && contains(captured,'h = 8;'));"
            }
        },
        BuiltinExample {
            id: "example-2",
            title: "Display selected lines",
            program: "dbtype myscript.m 3:8",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
                id: BuiltinExampleFixtureId {
                    local_name: "dbtype-2"
                },
                root: BuiltinFilesystemRoot::IsolatedWorkspace,
                entries: &[BuiltinFilesystemEntry::File {
                        relative_path: "myscript.m",
                        content: BuiltinFixtureContent::Utf8("a = 1;\nb = 2;\nc = 3;\nd = 4;\ne = 5;\nf = 6;\ng = 7;\nh = 8;\n")
                    }
                ]
            }
            ),
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "captured=evalc('dbtype myscript.m 3:8'); assert(contains(captured,'c = 3;') && contains(captured,'h = 8;') && ~contains(captured,'a = 1;'));"
            }
        }
    ],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
            question: "Can `dbtype` display Rust builtin source?",
            answer: "No. It displays MATLAB source files found by the RunMat filesystem/search path."
        }
    ],
    links: &[BuiltinDocumentationLink {
            label: "which",
            target: BuiltinDocumentationLinkTarget::Documentation("./which")
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
