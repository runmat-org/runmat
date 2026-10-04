use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("fileread"), slug: Some("fileread"),
    summary: "Read an entire file into a character row.",
    description: "`fileread(filename)` reads a host text file and returns its decoded content as a 1-by-N character array. An optional encoding can be passed positionally or as an `Encoding` name-value pair; invalid controls fail before file access.",
    keywords: &["fileread", "io", "read file", "text file", "character vector"],
    related: &["fopen", "fread", "string", "strlength", "filewrite"],
    sections: &[
        BuiltinDocumentationSection { heading: "Paths, encoding, and output", paragraphs: &[
            "The filename must be a scalar string or one-row character vector; a string array must have exactly one element. Relative paths use the RunMat process working directory. Bare-name MATLAB search-path lookup is not implemented. A missing or unreadable file raises a filesystem error.",
            "An empty file yields a 1-by-0 character row. CR, LF, and CRLF line endings are preserved. The default `auto` decoder tries UTF-8 and, on invalid UTF-8, maps each byte to its corresponding extended-ASCII code point. Explicit `utf-8` and `ascii` report incompatible bytes, while `latin1` and the RunMat `raw`/`bytes` alias map bytes one-for-one.",
            "`fileread(filename, encoding)` and `fileread(filename, 'Encoding', encoding)` accept auto, utf-8, ascii, latin1, and raw labels with documented aliases. The input is host text; numeric or provider-resident filename and encoding controls are not gathered and are rejected before filesystem effects.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and resource ownership", paragraphs: &[
            "The installed filesystem service supplies the bytes; the result is host character storage. There is no GPU kernel or fusion path. Browser execution requires a real installed file service and appropriate permissions.",
            "The current reader materializes the file bytes before decoding, so large files can require substantial host memory in addition to the returned character array.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "whole-file", title: "Read the entire file", program: "text = fileread('license.txt');\ntext", display_output: Some("text = Terms\nLine two\n"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "license" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: &[BuiltinFilesystemEntry::File { relative_path: "license.txt", content: BuiltinFixtureContent::Utf8("Terms\nLine two\n") }] }), requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(text, sprintf('Terms\\nLine two\\n')));" } },
        BuiltinExample { id: "relative-path", title: "Read a relative path", program: "text = fileread('data/config.json');\ntext", display_output: Some("text = {\"enabled\":true}"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "relative" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: &[BuiltinFilesystemEntry::Directory { relative_path: "data" }, BuiltinFilesystemEntry::File { relative_path: "data/config.json", content: BuiltinFixtureContent::Utf8("{\"enabled\":true}") }] }), requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(text, '{\"enabled\":true}'));" } },
        BuiltinExample { id: "extended-bytes", title: "Preserve extended ASCII bytes", program: "text = fileread('high_ascii.txt', 'raw');\ncodes = double(text);\ncodes", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "raw" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: &[BuiltinFilesystemEntry::File { relative_path: "high_ascii.txt", content: BuiltinFixtureContent::Bytes(&[65, 233, 255]) }] }), requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(codes, [65 233 255]));" } },
        BuiltinExample { id: "string-conversion", title: "Convert the result to a string scalar", program: "raw = fileread('readme.txt');\ndoc = string(raw);\ndoc", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "string" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: &[BuiltinFilesystemEntry::File { relative_path: "readme.txt", content: BuiltinFixtureContent::Utf8("RunMat documentation") }] }), requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(raw, 'RunMat documentation')); assert(strlength(doc) == 20);" } },
        BuiltinExample { id: "explicit-utf8", title: "Decode UTF-8 explicitly", program: "text = fileread('data/report.txt', 'Encoding', 'utf-8');\ntext", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "utf8" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: &[BuiltinFilesystemEntry::Directory { relative_path: "data" }, BuiltinFilesystemEntry::File { relative_path: "data/report.txt", content: BuiltinFixtureContent::Utf8("café\n") }] }), requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(text, sprintf('café\\n')));" } },
        BuiltinExample { id: "missing", title: "Report a missing file", program: "fileread('missing.txt');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::ExpectedError { identifier: "MATLAB:fileread:IoFailure" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What does `fileread` return?", answer: "A 1-by-N character vector, including a 1-by-0 row for an empty file. Use `string(...)` to convert it." },
        BuiltinDocumentationFaq { question: "Does it change line endings?", answer: "No. Original CR, LF, and CRLF characters remain in the returned text." },
        BuiltinDocumentationFaq { question: "Can it read raw bytes?", answer: "The RunMat raw/bytes encoding alias maps each input byte to one character code point; the compatible primary form is text-oriented." },
        BuiltinDocumentationFaq { question: "How does auto decoding work?", answer: "It tries UTF-8 first, then maps each byte to an extended-ASCII code point if UTF-8 is invalid. Explicit UTF-8 and ASCII reject incompatible input." },
        BuiltinDocumentationFaq { question: "How do relative paths resolve?", answer: "Against the process working directory, not through bare-name MATLAB search-path lookup." },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "fopen",
            target: BuiltinDocumentationLinkTarget::Builtin("fopen"),
        },
        BuiltinDocumentationLink {
            label: "fread",
            target: BuiltinDocumentationLinkTarget::Builtin("fread"),
        },
        BuiltinDocumentationLink {
            label: "string",
            target: BuiltinDocumentationLinkTarget::Builtin("string"),
        },
        BuiltinDocumentationLink {
            label: "strlength",
            target: BuiltinDocumentationLinkTarget::Builtin("strlength"),
        },
        BuiltinDocumentationLink {
            label: "fclose",
            target: BuiltinDocumentationLinkTarget::Builtin("fclose"),
        },
        BuiltinDocumentationLink {
            label: "feof",
            target: BuiltinDocumentationLinkTarget::Builtin("feof"),
        },
        BuiltinDocumentationLink {
            label: "fgets",
            target: BuiltinDocumentationLinkTarget::Builtin("fgets"),
        },
        BuiltinDocumentationLink {
            label: "filewrite",
            target: BuiltinDocumentationLinkTarget::Builtin("filewrite"),
        },
        BuiltinDocumentationLink {
            label: "fprintf",
            target: BuiltinDocumentationLinkTarget::Builtin("fprintf"),
        },
        BuiltinDocumentationLink {
            label: "frewind",
            target: BuiltinDocumentationLinkTarget::Builtin("frewind"),
        },
        BuiltinDocumentationLink {
            label: "fwrite",
            target: BuiltinDocumentationLinkTarget::Builtin("fwrite"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "File read and decode owner", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/filetext/fileread.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Encoding and content", location: "builtins::io::filetext::fileread::tests::fileread_reads_text_file" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
