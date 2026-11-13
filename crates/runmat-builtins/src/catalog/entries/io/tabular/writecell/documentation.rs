use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("writecell"),
    slug: Some("writecell"),
    summary: "Write heterogeneous cell arrays to delimited text or spreadsheet files.",
    description: "`writecell(C, filename)` serializes a 2-D cell array to delimited text or an Excel-compatible spreadsheet, preserving mixed numeric, logical, text, and empty cells.",
    keywords: &["writecell", "csv", "xlsx", "xls", "cell array", "delimited text", "spreadsheet", "append", "quote strings"],
    related: &["writematrix", "readmatrix", "xlsread", "csvwrite", "dlmwrite"],
    sections: &[
        BuiltinDocumentationSection { heading: "Cell content and output formats", paragraphs: &[
            "Delimited numeric text uses general formatting with 15 significant digits for double storage and 7 for single storage. Logical fields use 1 or 0; typed integers retain exact decimal values.",
            "Accepts 2-D cell arrays of numeric scalars, logical scalars, string scalars, row character vectors, and empty numeric/logical arrays. Scalar values from all eight native integer classes serialize from exact storage. Nested cells, complex values, and nonscalar member arrays produce errors instead of silent flattening.",
            "The filename extension selects the file type: `.xlsx` and `.xlsm` use OOXML spreadsheet output; `.xls`, `.xlsb`, and `.ods` are unsupported. Other extensions select delimited text unless `'FileType'` overrides the choice.",
            "Delimited text defaults to comma for `.csv`, tab for `.tsv` and `.tab`, whitespace for `.txt`, `.dat`, and `.dlm`, and comma otherwise. `'Delimiter'`, `'QuoteStrings'`, `'WriteMode'`, and `'LineEnding'` control text output. Strings are quoted by default with doubled embedded quotes; `'QuoteStrings', false` writes text verbatim. `'WriteMode','append'` appends rows to text files.",
            "Spreadsheet output writes one worksheet. `'Sheet'` accepts a worksheet name or positive one-based numeric index from any native integer class; `'Range'` selects the top-left placement. Spreadsheet append is rejected rather than partially mutating a workbook.",
            "MATLAB documents a statement form. A requested bytes-written output is a named RunMat extension; ordinary statement calls do not expose it.",
        ] },
        BuiltinDocumentationSection { heading: "Residency and filesystem effects", paragraphs: &[
            "Serialization runs on the host and is a fusion boundary. Automatically resident cell members gather through their owners; direct explicit `gpuArray` input requires RunMat mode. No provider kernel or resident output is advertised.",
            "The selected filesystem provider, relative directory and advertised lease are retained across preparation and writing. Overwrite uses a temporary sibling and a provider-owned replacement operation; text append writes to the selected provider. A failure or cancellation may leave partial output or a temporary file. Complete-call finite codec and formatting memory evidence is not certified here.",
            "Historical programs and captions are retained below; the runnable examples use isolated destination fixtures with content assertions:",
            "Historical example 1 (Write heterogeneous cells to CSV):\nC = {1.5, \"alpha\"; true, []};\nwritecell(C, \"out.csv\")",
            "Historical example 2 (Append rows to a pipe-delimited text file):\nwritecell({\"a,b\", 2}, \"report.txt\", 'Delimiter', '|', 'QuoteStrings', false);\nwritecell({\"tail\", 3}, \"report.txt\", 'Delimiter', '|', 'QuoteStrings', false, 'WriteMode', 'append')",
            "Historical example 3 (Write to a spreadsheet sheet and range):\nwritecell({\"Voltage\", 1.5, true}, \"measurements.xlsx\", 'Sheet', 'Data', 'Range', 'B2')",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "heterogeneous-csv", title: "Write heterogeneous cells to CSV", program: "C = {1.5,\"alpha\";true,[]};\nwritecell(C,'out.csv');\ncontents = fileread('out.csv')", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "heterogeneous-csv" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: &[BuiltinFilesystemEntry::File { relative_path: "out.csv", content: BuiltinFixtureContent::Utf8("old-data\n") }] }), requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(contains(contents,'1.5')); assert(contains(contents,'alpha')); assert(~contains(contents,'old-data'));" } },
        BuiltinExample { id: "pipe-append", title: "Append rows to a pipe-delimited text file", program: "writecell({\"tail\",3},'report.txt','Delimiter','|','QuoteStrings',false,'WriteMode','append');\ncontents = fileread('report.txt')", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "pipe-append" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: &[BuiltinFilesystemEntry::File { relative_path: "report.txt", content: BuiltinFixtureContent::Utf8("a,b|2\n") }] }), requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(contains(contents,'a,b|2')); assert(contains(contents,'tail|3'));" } },
        BuiltinExample { id: "spreadsheet-range", title: "Write to a spreadsheet sheet and range", program: "writecell({\"Voltage\",1.5,true},'measurements.xlsx','Sheet','Data','Range','B2');\n[num,txt,raw] = xlsread('measurements.xlsx','Data','B2:D2')", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "spreadsheet-range" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: &[BuiltinFilesystemEntry::File { relative_path: "measurements.xlsx", content: BuiltinFixtureContent::Bytes(&[]) }] }), requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(contains(raw{1,1},'Voltage')); assert(raw{1,2}==1.5); assert(raw{1,3}==1);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does `writecell` support heterogeneous data?", answer: "Yes. Mixed numeric, logical, text, and empty cells are the intended tabular export form." },
        BuiltinDocumentationFaq { question: "Can `writecell` write Excel files?", answer: "RunMat writes OOXML `.xlsx` and `.xlsm` workbooks with worksheet names, numeric sheet indices and top-left ranges; `.xls`, `.xlsb`, and `.ods` are unsupported." },
        BuiltinDocumentationFaq { question: "How are strings quoted in CSV output?", answer: "Strings are quoted by default and embedded double quotes are doubled. Set `'QuoteStrings', false` to write verbatim." },
        BuiltinDocumentationFaq { question: "Can I append to an existing `.xlsx` file?", answer: "Not yet. Text append works; spreadsheet append raises an option error before mutation." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "MATLAB language extensions", target: BuiltinDocumentationLinkTarget::Documentation("/docs/runtime/getting-started/language-extensions") },
        BuiltinDocumentationLink { label: "writematrix", target: BuiltinDocumentationLinkTarget::Builtin("writematrix") },
        BuiltinDocumentationLink { label: "readmatrix", target: BuiltinDocumentationLinkTarget::Builtin("readmatrix") },
        BuiltinDocumentationLink { label: "xlsread", target: BuiltinDocumentationLinkTarget::Builtin("xlsread") },
        BuiltinDocumentationLink { label: "csvwrite", target: BuiltinDocumentationLinkTarget::Builtin("csvwrite") },
        BuiltinDocumentationLink { label: "dlmwrite", target: BuiltinDocumentationLinkTarget::Builtin("dlmwrite") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "writecell runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/tabular/writecell.rs") },
            BuiltinDocumentationLink { label: "Selected provider writer", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/tabular/write_owner.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Cell content, delimited output, spreadsheet sheet/range and rejection", location: "builtins::io::tabular::writecell::tests" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
