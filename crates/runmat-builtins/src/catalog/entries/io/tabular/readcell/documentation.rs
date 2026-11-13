use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("readcell"),
    slug: Some("readcell"),
    summary: "Read tabular data into a cell array.",
    description: "`readcell(filename, ...)` imports text or spreadsheet data directly into a cell array without consuming detected table headers.",
    keywords: &[
        "readcell",
        "cell",
        "readtable",
        "csv",
        "spreadsheet",
    ],
    related: &[
        "readtable",
        "table2cell",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "Text and spreadsheet options accepted by `readtable` are reused where they apply.",
                "A typed-integer `NumHeaderLines` is accepted as the documented positive-integer structural control and is read exactly from every native integer class.",
                "Typed-integer values for controls whose public datatype contract excludes native integers are RunMat-only extensions and are rejected in MATLAB compatibility mode before gathering or file access.",
                "Rows are padded to a rectangular cell array.",
                "Delimited text cells are converted to double or logical scalars when they parse cleanly; otherwise they remain strings.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &[
                "File import runs on the host.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Read cells",
            program: "C = readcell('data.csv')",
            display_output: Some("C = cell array"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
                id: BuiltinExampleFixtureId {
                    local_name: "readcell-1",
                },
                root: BuiltinFilesystemRoot::IsolatedWorkspace,
                entries: &[
                    BuiltinFilesystemEntry::File {
                        relative_path: "data.csv",
                        content: BuiltinFixtureContent::Utf8("1,true,alpha\n2,false,beta\n")
                    },
                ],
            } ),
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(iscell(C)); assert(isequal(size(C), [2 3])); assert(C{1,1} == 1); assert(C{2,1} == 2); assert(C{1,2} == true); assert(C{2,2} == false); assert(strcmp(C{1,3}, \"alpha\")); assert(strcmp(C{2,3}, \"beta\"));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink {
            label: "readtable",
            target: BuiltinDocumentationLinkTarget::Builtin("readtable")
        },
        BuiltinDocumentationLink {
            label: "table2cell",
            target: BuiltinDocumentationLinkTarget::Builtin("table2cell")
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "`crates/runmat-runtime/src/builtins/table/mod.rs`",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/mod.rs")
            },
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::table::tests",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
