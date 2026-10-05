use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("xlsread"),
    slug: Some("xlsread"),
    summary: "Read numeric, text, and raw data from spreadsheet files.",
    description: "`xlsread` imports spreadsheet data using MATLAB's legacy call forms. The first output is a dense double matrix; optional second and third outputs return text cells and raw cells.",
    keywords: &["xlsread", "xls", "xlsx", "spreadsheet", "excel", "numeric import"],
    related: &["readmatrix", "readtable", "spreadsheetImportOptions", "xlswrite", "writecell"],
    sections: &[
        BuiltinDocumentationSection { heading: "Forms and output", paragraphs: &[
            "`xlsread(filename)` reads the first worksheet. If the filename has no extension, `.xlsx` is appended.",
            "`xlsread(filename, range)` reads an Excel A1 range such as `\"C4:C19\"` from the first worksheet.",
            "`xlsread(filename, sheet)` selects a worksheet by name or one-based numeric index.",
            "A numeric worksheet index may use any native integer class and is decoded exactly before selecting the worksheet.",
            "`xlsread(filename, sheet, range)` combines worksheet and range selection.",
            "The legacy `'-basic'` mode flag is accepted as a no-op because RunMat always uses its host spreadsheet parser rather than Excel COM automation.",
            "The numeric output trims leading and trailing rows/columns that contain no numeric cells, then fills nonnumeric cells inside the numeric rectangle with `NaN`.",
            "`[num, txt, raw] = xlsread(...)` returns text cells in `txt` and mixed number/string/logical cells in `raw`.",
            "RunMat mode accepts a numeric row and column vector in the range position and accepts explicit `gpuArray` selectors; both are gated before file access.",
            "Excel callback/process-function modes are not supported and raise a descriptive argument error.",
            "The historical illustrations are retained: xlsread('diod.xlsx','C4:C19') followed by plot(V); xlsread('measurements.xlsx','Data','D4:D19'); and [num,txt,raw] = xlsread('measurements.xlsx','A1:C10'). The executable examples below create isolated workbooks first.",
        ] },
        BuiltinDocumentationSection { heading: "Filesystem and execution", paragraphs: &[
            "The selected filesystem provider, current directory and home expansion are captured before selectors can suspend. The same provider opens the workbook; results are host-resident and form a fusion boundary. Explicit resident selectors require the named RunMat extension and are gathered by their owning provider.",
            "Workbook input is read in bounded chunks with cancellation and lease checks. Synchronous spreadsheet decompression and complete-call finite memory accounting remain shared codec/resource requirements; a file-size cap is not a whole-call memory certificate.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "column", title: "Read one spreadsheet column", program: "xlswrite('diod.xlsx',[zeros(20,2),(1:20)']);\nV = xlsread('diod.xlsx','C4:C19');\nV", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(V,(4:19)'));" } },
        BuiltinExample { id: "named-sheet", title: "Read a named worksheet and range", program: "xlswrite('measurements.xlsx',[zeros(20,3),(1:20)'],'Data');\nI = xlsread('measurements.xlsx','Data','D4:D19');\nI", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(I,(4:19)'));" } },
        BuiltinExample { id: "three-outputs", title: "Capture numeric, text, and raw outputs", program: "xlswrite('measurements.xlsx',{'Name','Value';'A',1;'B',2});\n[num,txt,raw] = xlsread('measurements.xlsx','A1:B3');\nnum", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(num,[1;2])); assert(iscell(txt)); assert(iscell(raw));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Which spreadsheet formats are supported?", answer: "RunMat uses the same spreadsheet reader as `readtable`, covering common Excel/OpenDocument formats handled by the runtime parser, including `.xls`, `.xlsx`, `.xlsm`, `.xlsb`, and `.ods`." },
        BuiltinDocumentationFaq { question: "Does `xlsread` use Microsoft Excel?", answer: "No. RunMat reads spreadsheets directly on the host. The `'-basic'` compatibility flag is accepted, but COM automation and Excel callback processing are not part of RunMat's runtime model." },
        BuiltinDocumentationFaq { question: "What should I use for new code?", answer: "`xlsread` exists for legacy MATLAB compatibility. For new mixed-type imports, prefer `readtable`; for numeric matrices, `readmatrix` is the modern MATLAB API." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "MATLAB language extensions", target: BuiltinDocumentationLinkTarget::External("/docs/runtime/getting-started/language-extensions") },
        BuiltinDocumentationLink { label: "readmatrix", target: BuiltinDocumentationLinkTarget::Builtin("readmatrix") },
        BuiltinDocumentationLink { label: "readtable", target: BuiltinDocumentationLinkTarget::Builtin("readtable") },
        BuiltinDocumentationLink { label: "spreadsheetImportOptions", target: BuiltinDocumentationLinkTarget::Builtin("spreadsheetImportOptions") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Spreadsheet reader", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/tabular/xlsread.rs") },
            BuiltinDocumentationLink { label: "Captured filesystem owner", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/tabular/path.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Spreadsheet formats, selectors, output counts and zip bounds", location: "builtins::io::tabular::xlsread::tests" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
