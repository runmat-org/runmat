use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("xlswrite"),
    slug: Some("xlswrite"),
    summary: "Write numeric, logical, string, character, or cell data to Excel-compatible spreadsheet files.",
    description: "`xlswrite` provides MATLAB's legacy spreadsheet export forms for older scripts. It writes an OOXML workbook payload for `.xlsx` and legacy `.xls` filenames, accepts worksheet and range selectors, preserves supported cells in workbooks previously created by RunMat `xlswrite`, and returns status/message outputs when requested.",
    keywords: &["xlswrite", "xls", "xlsx", "spreadsheet", "excel", "legacy export"],
    related: &["xlsread", "writematrix", "writecell", "readtable"],
    sections: &[
        BuiltinDocumentationSection { heading: "Files, sheets, and ranges", paragraphs: &[
            "`xlswrite(filename, A)` writes to the first worksheet starting at A1; a filename without an extension gains `.xlsx`. A third argument selects a worksheet by name or positive one-based numeric index, or selects an A1 range such as `B2` on the default worksheet. The four-argument form combines sheet and range.",
            "A1 ranges may use absolute markers such as `$B$2:$C$3`; malformed endpoints are rejected. Numeric sheet ordinals create missing `SheetN` placeholders so `xlsread(filename, N, ...)` addresses the same worksheet.",
            "RunMat mode accepts an exact native-integer row/column vector as the range and explicit `gpuArray` input. Both extensions are gated before file access. The legacy `'-basic'` flag is accepted as a no-op because writing does not use Excel COM automation.",
            "`.xls` filenames contain the same OOXML payload as `.xlsx`, readable by RunMat's content-based spreadsheet reader. Existing RunMat-marked workbooks preserve supported sheets and cells outside the overlay. Existing unmarked Excel workbooks are rejected to avoid silently losing formulas, macros, styles, charts, images, comments, or metadata.",
        ] },
        BuiltinDocumentationSection { heading: "Values, residency, and status", paragraphs: &[
            "Numeric, sparse, logical, string, character, and heterogeneous cell inputs are supported. All eight native integer classes serialize from exact storage; cell contents must be scalar spreadsheet values.",
            "The operation writes files on the host and is a fusion boundary. Automatically resident values gather through their owning provider; explicit device intent requires RunMat mode. Outputs are host status/message values, not resident arrays.",
            "`status = xlswrite(...)` returns true on success. `[status, message] = xlswrite(...)` captures a known pre-publication failure as false and a struct with `message` and `identifier` fields. A statement call raises its error instead of returning status. If publication is known to have committed or its outcome cannot be determined, the call raises an error even when status outputs were requested; do not automatically retry it.",
            "A selected filesystem provider and directory are retained across materialization. Complete-call finite workbook/codec memory evidence is not certified for a constrained host budget.",
            "Historical example (Write a numeric matrix):\nstatus = xlswrite('measurements.xlsx', [1 2 3; 4 5 6]);",
            "Historical example (Write to a named worksheet and range):\nstatus = xlswrite('measurements.xlsx', {'Voltage', 1.5, true}, 'Data', 'B2');",
            "Historical example (Capture a failure message):\n[status, message] = xlswrite('out.xlsx', 1+2i);\nif ~status\n    disp(message.message)\nend",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "numeric-matrix", title: "Write a numeric matrix", program: "status = xlswrite('sheets/measurements.xlsx',[1 2 3;4 5 6]);\nactual = xlsread('sheets/measurements.xlsx')", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "numeric-matrix" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: &[BuiltinFilesystemEntry::Directory { relative_path: "sheets" }] }), requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(status); assert(isequal(actual,[1 2 3;4 5 6]));" } },
        BuiltinExample { id: "named-sheet-range", title: "Write to a named worksheet and range", program: "status = xlswrite('sheets/measurements.xlsx',{'Voltage',1.5,true},'Data','B2');\n[num,txt,raw] = xlsread('sheets/measurements.xlsx','Data','B2:D2')", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "named-sheet-range" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: &[BuiltinFilesystemEntry::Directory { relative_path: "sheets" }] }), requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(status); assert(strcmp(raw{1,1},'Voltage')); assert(raw{1,2}==1.5); assert(raw{1,3}==1);" } },
        BuiltinExample { id: "captured-failure", title: "Capture a failure message", program: "[status,message] = xlswrite('out.xlsx',1+2i);\nif ~status\n    disp(message.message)\nend", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "captured-failure" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: &[BuiltinFilesystemEntry::File { relative_path: "out.xlsx", content: BuiltinFixtureContent::Bytes(&[]) }] }), requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(~status); assert(strcmp(message.identifier,'RunMat:xlswrite:Data')); assert(~isempty(message.message));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does `xlswrite` use Microsoft Excel?", answer: "No. RunMat writes workbook files directly on the host. The `'-basic'` flag is accepted for compatibility, but COM automation is not used." },
        BuiltinDocumentationFaq { question: "What happens with `.xls` filenames?", answer: "RunMat accepts legacy `.xls` filenames and writes the same OOXML payload used for `.xlsx`, readable by its content-based spreadsheet reader." },
        BuiltinDocumentationFaq { question: "Can `xlswrite` update an existing Excel workbook created elsewhere?", answer: "No. It rejects unmarked workbooks rather than rewriting them as value-only workbooks and losing formulas, macros, formatting, charts, images, comments, or metadata." },
        BuiltinDocumentationFaq { question: "What should I use for new code?", answer: "`xlswrite` exists for legacy MATLAB compatibility. For new code, prefer `writematrix`, `writecell`, or table-oriented export APIs where available." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "MATLAB language extensions", target: BuiltinDocumentationLinkTarget::Documentation("/docs/runtime/getting-started/language-extensions") },
        BuiltinDocumentationLink { label: "xlsread", target: BuiltinDocumentationLinkTarget::Builtin("xlsread") },
        BuiltinDocumentationLink { label: "writematrix", target: BuiltinDocumentationLinkTarget::Builtin("writematrix") },
        BuiltinDocumentationLink { label: "writecell", target: BuiltinDocumentationLinkTarget::Builtin("writecell") },
        BuiltinDocumentationLink { label: "readtable", target: BuiltinDocumentationLinkTarget::Builtin("readtable") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "xlswrite runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/tabular/xlswrite.rs") },
            BuiltinDocumentationLink { label: "Selected-provider writer", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/tabular/write_owner.rs") },
            BuiltinDocumentationLink { label: "Publication outcome", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/tabular/write_owner/publication.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Workbook overlay, sheet/range, status capture, and exact integer cells", location: "builtins::io::tabular::xlswrite::tests" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
