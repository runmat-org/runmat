use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("detectImportOptions"),
    slug: Some("detectImportOptions"),
    summary: "Inspect text or spreadsheet data and create import options for tabular readers.",
    description: "`opts = detectImportOptions(filename)` samples a delimited text or spreadsheet file and returns an editable scalar struct accepted by `readtable` and, for numeric text imports, `readmatrix`.",
    keywords: &[
        "detectImportOptions",
        "readtable",
        "readmatrix",
        "csv",
        "tsv",
        "spreadsheet",
        "VariableTypes",
        "VariableNames",
    ],
    related: &[
        "readtable",
        "readmatrix",
        "spreadsheetImportOptions",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "Detects text delimiters from comma, tab, semicolon, pipe, and whitespace-separated records unless `Delimiter` is supplied.",
                "Infers variable names from a non-data header row and sets `NumHeaderLines` so readers import only data rows.",
                "Infers `VariableTypes` as `double`, `logical`, `datetime`, or text (`string`/`char`) using the same token rules as `readtable`.",
                "Honors the implemented text/spreadsheet override subset, including `FileType`, `Delimiter`, `NumHeaderLines`, `Range`, `DataRange`, `Sheet`, `ExpectedNumVariables`, `VariableNamesLine`, `VariableNames`, `VariableTypes`, `ReadVariableNames`, `ReadRowNames`, `TreatAsMissing`, `VariableNamingRule`, `TextType`, `DatetimeType`, `EmptyLineRule`, and `Encoding`.",
                "Integer `Range`, `DataRange`, `Sheet`, `ExpectedNumVariables`, and `VariableNamesLine` controls are parsed exactly from all eight integer classes. `ExpectedNumVariables` and `Sheet` must be positive; boolean controls accept only logical values or exact numeric zero/one.",
                "Numeric `DataRange` currently supports a positive scalar start row or one 1-by-2 increasing row interval. Textual cell references are supported for detected-option replay; multiple numeric Nx2 intervals remain a recorded general gap.",
                "Typed-integer `NumHeaderLines` is a RunMat-only extension. Compatibility mode rejects it before provider or file access; scalar single and double remain public inputs.",
                "For spreadsheet files, uses the same sheet/range reader as `readtable` and emits spreadsheet-compatible option fields.",
                "The returned struct intentionally contains fields understood by current RunMat readers. Concrete MATLAB import-options classes and JSON, XML, Word, HTML, and archive option families remain recorded general gaps.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &[
                "Runs on the host. Resident arguments, including nested option values, are rejected before provider or file access.",
                "`detectImportOptions` performs file inspection on the host and returns host metadata. It does not create GPU-resident arrays.",
                "No acceleration provider hooks are used. Resident arguments are rejected without a download.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Detect And Reuse CSV Import Options",
            program: "opts = detectImportOptions(\"measurements.csv\");\nT = readtable(\"measurements.csv\", opts);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
                id: BuiltinExampleFixtureId {
                    local_name: "detectimportoptions-1",
                },
                root: BuiltinFilesystemRoot::IsolatedWorkspace,
                entries: &[
                    BuiltinFilesystemEntry::File {
                        relative_path: "measurements.csv",
                        content: BuiltinFixtureContent::Utf8("Count,Value\n1,10\n2,20\n")
                    },
                ],
            } ),
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isstruct(opts)); assert(isequal(T.Count, [1; 2])); assert(isequal(T.Value, [10; 20]));",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Override Detection Settings",
            program: "opts = detectImportOptions(\"log.txt\", 'Delimiter', '|', 'NumHeaderLines', 2);\nopts.VariableTypes(2) = \"double\";\nT = readtable(\"log.txt\", opts);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
                id: BuiltinExampleFixtureId {
                    local_name: "detectimportoptions-2",
                },
                root: BuiltinFilesystemRoot::IsolatedWorkspace,
                entries: &[
                    BuiltinFilesystemEntry::File {
                        relative_path: "log.txt",
                        content: BuiltinFixtureContent::Utf8("Sensor log\nGenerated by fixture\nName|Value\nalpha|1\nbeta|2\n")
                    },
                ],
            } ),
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(strcmp(opts.Delimiter, \"|\")); assert(strcmp(class(T.Value), \"double\")); assert(isequal(T.Value, [1; 2]));",
            },
        },
        BuiltinExample {
            id: "retained-3",
            title: "Use Options With Numeric Imports",
            program: "opts = detectImportOptions(\"values.csv\");\nA = readmatrix(\"values.csv\", opts);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
                id: BuiltinExampleFixtureId {
                    local_name: "detectimportoptions-3",
                },
                root: BuiltinFilesystemRoot::IsolatedWorkspace,
                entries: &[
                    BuiltinFilesystemEntry::File {
                        relative_path: "values.csv",
                        content: BuiltinFixtureContent::Utf8("1,2\n3,4\n")
                    },
                ],
            } ),
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(A, [1 2; 3 4]));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Is the returned value a MATLAB import-options object?",
            answer: "RunMat represents import options as a scalar struct so field assignment and reader consumption work naturally in the current runtime.",
        },
        BuiltinDocumentationFaq {
            question: "Can I use the same options with `readmatrix`?",
            answer: "Yes for numeric text imports. `readmatrix` consumes fields such as `Delimiter`, `NumHeaderLines`, `Range`, and `TreatAsMissing` and ignores table-specific fields.",
        },
        BuiltinDocumentationFaq {
            question: "Does detection read the whole file?",
            answer: "RunMat currently reads the file into host memory and samples the parsed rows, matching the existing `readtable` import pipeline.",
        },
        BuiltinDocumentationFaq {
            question: "Can detection options be GPU-resident?",
            answer: "No. Resident path or option values, including nested values, are rejected before provider or file access.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "readtable",
            target: BuiltinDocumentationLinkTarget::Builtin("readtable")
        },
        BuiltinDocumentationLink {
            label: "readmatrix",
            target: BuiltinDocumentationLinkTarget::Builtin("readmatrix")
        },
        BuiltinDocumentationLink {
            label: "spreadsheetImportOptions",
            target: BuiltinDocumentationLinkTarget::Builtin("spreadsheetImportOptions")
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "`crates/runmat-runtime/src/builtins/table/builtins/io.rs`",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/builtins/io.rs")
            },
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::table::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "integration 1",
                location: "builtins::table::tests::detect_import_options_infers_text_delimiter_names_and_types",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "integration 2",
                location: "builtins::table::tests::detect_import_options_struct_can_drive_readmatrix",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "integration 3",
                location: "builtins::table::tests::detect_import_options_honors_overrides_and_range",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
