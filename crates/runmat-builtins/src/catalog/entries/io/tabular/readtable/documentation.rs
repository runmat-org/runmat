use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("readtable"),
    slug: Some("readtable"),
    summary: "Import tabular text or spreadsheet data into a table.",
    description: "`readtable(filename, ...)` imports delimited text or spreadsheet data into a table using name-value controls or an import-options structure.",
    keywords: &[
        "readtable",
        "table",
        "csv",
        "spreadsheet",
        "VariableTypes",
        "VariableNamesLine",
    ],
    related: &[
        "readcell",
        "readtimetable",
        "detectImportOptions",
        "spreadsheetImportOptions",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "Supports inferred or explicit variable names, missing tokens, text type, row names, sheet and range selection, and import-options structures created by `detectImportOptions` or `spreadsheetImportOptions`.",
                "Documented integer `VariableTypes` preserve all eight native classes. Decimal fields are parsed directly into the requested integer class, including exact `int64` and `uint64` values above `flintmax`, with class-specific saturation for out-of-range values.",
                "The documented `VariableNamesLine` structural control accepts all eight integer classes and is bounds-checked exactly.",
                "Typed-integer values for implemented controls whose public datatype table excludes native integers are RunMat-only extensions. MATLAB compatibility mode rejects those forms before gathering or opening the file; ordinary residency still gathers transparently for host import.",
                "Imported table variables retain their double, single, integer, logical, text, datetime, or duration classes and values.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &[
                "File import and table construction run on the host. Ordinary resident arguments gather transparently before file access.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Import exact unsigned 64-bit values",
            program: "opts = spreadsheetImportOptions('NumVariables', 1);\nopts.VariableNames = {'Counter'};\nopts.VariableTypes = {'uint64'};\nT = readtable('counters.xlsx', opts)",
            display_output: Some("% T.Counter is a native uint64 table variable."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: super::super::documentation_fixtures::COUNTERS,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(strcmp(class(T.Counter), \"uint64\")); assert(isequal(size(T.Counter), [2 1])); assert(T.Counter(1) == bitshift(uint64(1), 53) + uint64(1)); assert(T.Counter(2) == intmax(\"uint64\"));",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Select the variable-name line with a native integer",
            program: "T = readtable('report.csv', 'VariableNamesLine', uint8(2))",
            display_output: Some("% The second input line supplies table variable names."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
                id: BuiltinExampleFixtureId {
                    local_name: "readtable-2",
                },
                root: BuiltinFilesystemRoot::IsolatedWorkspace,
                entries: &[
                    BuiltinFilesystemEntry::File {
                        relative_path: "report.csv",
                        content: BuiltinFixtureContent::Utf8("Report generated\nCount,Value\n1,10\n2,20\n")
                    },
                ],
            } ),
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(T.Properties.VariableNames, [\"Count\" \"Value\"])); assert(isequal(T.Count, [1; 2])); assert(isequal(T.Value, [10; 20]));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink {
            label: "readcell",
            target: BuiltinDocumentationLinkTarget::Builtin("readcell")
        },
        BuiltinDocumentationLink {
            label: "readtimetable",
            target: BuiltinDocumentationLinkTarget::Builtin("readtimetable")
        },
        BuiltinDocumentationLink {
            label: "detectImportOptions",
            target: BuiltinDocumentationLinkTarget::Builtin("detectImportOptions")
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
                label: "integration",
                location: "builtins::table::tests::readtable_variable_types_preserve_all_exact_integer_classes",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
