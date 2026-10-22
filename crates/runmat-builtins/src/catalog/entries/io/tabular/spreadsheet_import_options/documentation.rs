use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("spreadsheetImportOptions"),
    slug: Some("spreadsheetImportOptions"),
    summary: "Create spreadsheet import options structs for `readtable`.",
    description: "`opts = spreadsheetImportOptions(...)` returns an editable import-options struct accepted by `readtable` for spreadsheet files. The struct supports MATLAB-style dot assignment before import, including data ranges, sheet selection, variable names, and variable types.",
    keywords: &[
        "spreadsheetImportOptions",
        "readtable",
        "spreadsheet",
        "xlsx",
        "xls",
        "DataRange",
        "VariableTypes",
    ],
    related: &[
        "readtable",
        "readmatrix",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "Accepts name-value options including `'NumVariables'`, `'VariableNames'`, `'VariableTypes'`, `'DataRange'`, `'Range'`, `'Sheet'`, `'ReadVariableNames'`, `'ReadRowNames'`, `'TreatAsMissing'`, `'VariableNamingRule'`, `'TextType'`, `'DatetimeType'`, and `'EmptyLineRule'`.",
                "`'NumVariables'` creates default `Var1`, `Var2`, ... names and `auto` variable types; edited `VariableNames` and `VariableTypes` are validated against that count.",
                "`DataRange` is treated as the spreadsheet cell range for `readtable`, matching the common MATLAB workflow `opts.DataRange = 'C2:C37'`.",
                "`VariableTypes` entries guide import coercion for double, single, all eight native integer classes, logical, string, char, cellstr, datetime, and duration columns. Integer text is parsed directly into authoritative storage, including exact 64-bit values.",
                "A typed-integer `NumVariables` is accepted as the documented positive scalar integer and must fit the host index domain. Typed integers used for other location controls are a RunMat-only extension and are rejected in MATLAB compatibility mode before metadata construction.",
                "Default options do not force a zero-column import; `readtable` infers the column count unless a positive `NumVariables` is supplied.",
                "Invalid option names, malformed ranges, unsupported sheets, and unparseable typed values raise descriptive `readtable` errors instead of silently changing the schema.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &[
                "Builds host-side import metadata. File I/O and parsing happen later when the options are passed to `readtable`.",
                "No GPU work is performed. Options are host metadata, and later `readtable` imports create host table variables.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Import A Spreadsheet Column With An Options Struct",
            program: "opts = spreadsheetImportOptions('NumVariables', 1);\nopts.DataRange = 'C2:C37';\nopts.VariableNames = {'Amount'};\nopts.VariableTypes = {'double'};\nT = readtable('data.xlsx', opts)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: super::super::documentation_fixtures::AMOUNTS,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isstruct(opts)); assert(isequal(T.Amount, (1:36)')); assert(strcmp(class(T.Amount), \"double\"));",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Select A Worksheet And Preserve Variable Names",
            program: "opts = spreadsheetImportOptions('Sheet', 'Data', 'VariableNamingRule', 'preserve');\nT = readtable('workbook.xlsx', opts)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: super::super::documentation_fixtures::WORKBOOK,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(strcmp(opts.Sheet, \"Data\")); assert(strcmp(opts.VariableNamingRule, \"preserve\")); assert(isequal(T.Properties.VariableNames, [\"Item Name\" \"Value\"])); assert(isequal(T.Value, [10; 20]));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Is the returned value an object?",
            answer: "RunMat represents import options as a scalar struct so MATLAB-style field assignment works naturally and `readtable` can consume the same value directly.",
        },
        BuiltinDocumentationFaq {
            question: "What happens if `VariableTypes` asks for a type that cannot parse a column?",
            answer: "`readtable` raises a typed import error for the offending non-missing value. It does not fall back to a string column when the schema explicitly requested another type.",
        },
        BuiltinDocumentationFaq {
            question: "Can I use the options with text files?",
            answer: "The constructor is spreadsheet-focused, but the struct fields that also apply to text imports are parsed by `readtable`; spreadsheet-only fields such as `Sheet` remain valid only for spreadsheet files.",
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
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "integration",
                location: "builtins::table::tests::readtable_consumes_spreadsheet_import_options_struct",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
