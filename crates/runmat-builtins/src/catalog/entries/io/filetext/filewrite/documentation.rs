use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("filewrite"),
    slug: Some("filewrite"),
    summary: "Write text or raw bytes with RunMat's native file-writing extension.",
    description: "`filewrite(filename, data)` writes text or raw bytes and returns the number of bytes written. It is a RunMat-native extension, not a MATLAB builtin; MATLAB-compatible mode rejects it before gathering data or changing the filesystem.",
    keywords: &["filewrite", "io", "write file", "text file", "append", "encoding"],
    related: &["fileread", "fwrite", "fprintf", "string", "fclose", "feof", "fgets", "fopen", "fread", "frewind"],
    sections: &[
        BuiltinDocumentationSection { heading: "Inputs and bytes", paragraphs: &[
            "The filename is a one-row character vector, string scalar, or one-element string array. Character vectors, string scalars, and string arrays are text data; string-array elements are joined in column-major order with newline separators. Numeric arrays and logical arrays are raw bytes, with logical false/true mapping to 0/1. All eight integer classes use exact integer storage and each value must be in 0..255; double values must pass integral-byte validation.",
            "The default write mode truncates the target. `'WriteMode','append'` adds bytes without inserting a newline. A missing parent directory is not created. The returned count is a host double scalar measuring bytes written, not characters.",
            "A positional encoding or `'Encoding',value` pair accepts auto, utf-8, ascii, latin1, and raw (with documented aliases). Auto and utf-8 encode text as UTF-8; ascii and latin1 reject characters outside their code pages. Numeric data already represents bytes, so encoding has no conversion effect except ASCII validation.",
        ] },
        BuiltinDocumentationSection { heading: "Host execution", paragraphs: &[
            "File I/O executes through the installed host filesystem. Provider-resident real numeric data is gathered through its owning provider before payload conversion; there is no filewrite GPU kernel or fusion path. Filename and option controls must already be host text, and known-invalid forms fail before data download or filesystem effects.",
            "The current writer prepares the complete payload in host memory before opening the destination. A resident value without its owning acceleration provider raises a gather-provider error; no GPU kernel is launched. Permission, missing-directory, write, and flush failures retain their underlying I/O cause. Browser use requires a real installed filesystem service.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "text", title: "Write text to a new file", program: "bytes = filewrite('notes.txt', 'Hello, RunMat!');\nbytes", display_output: Some("bytes = 14"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(bytes == 14); assert(strcmp(fileread('notes.txt'), 'Hello, RunMat!'));" } },
        BuiltinExample { id: "append", title: "Append without overwriting", program: "filewrite('log.txt', sprintf('First line\\n'));\nbytes = filewrite('log.txt', sprintf('Second line\\n'), 'WriteMode', 'append');\nbytes", display_output: Some("bytes = 12"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(bytes == 12); assert(strcmp(fileread('log.txt'), sprintf('First line\\nSecond line\\n')));" } },
        BuiltinExample { id: "latin1", title: "Select a text encoding", program: "bytes = filewrite('latin1.txt', ['E' 's' 'p' 'a' char(241) 'a'], 'Encoding', 'latin1');\nbytes", display_output: Some("bytes = 6"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(bytes == 6); assert(isequal(double(fileread('latin1.txt', 'raw')), [69 115 112 97 241 97]));" } },
        BuiltinExample { id: "raw", title: "Write raw numeric bytes", program: "payload = [1 2 3 255];\nbytes = filewrite('data.bin', payload, 'Encoding', 'raw');\nbytes", display_output: Some("bytes = 4"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(bytes == 4); assert(isequal(double(fileread('data.bin', 'raw')), [1 2 3 255]));" } },
        BuiltinExample { id: "strings", title: "Export a string array", program: "lines = [\"alpha\", \"beta\", \"gamma\"];\nbytes = filewrite('items.txt', lines);\nbytes", display_output: Some("bytes = 16"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(bytes == 16); assert(strcmp(fileread('items.txt'), sprintf('alpha\\nbeta\\ngamma')));" } },
        BuiltinExample { id: "ascii-error", title: "Reject non-ASCII characters in ASCII mode", program: "filewrite('ascii.txt', ['c' 'a' 'f' char(233)], 'Encoding', 'ascii');", display_output: Some("filewrite: failed to encode payload: character 'é' (U+00E9) cannot be encoded as ASCII"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::ExpectedError { identifier: "RunMat:filewrite:EncodeFailed" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What does `filewrite` return?", answer: "A double scalar counting bytes written. The output may be omitted when it is not needed." },
        BuiltinDocumentationFaq { question: "How are string arrays written?", answer: "Their elements are written in column-major order, separated by newlines. This is RunMat-native behavior." },
        BuiltinDocumentationFaq { question: "Is a newline added automatically?", answer: "Not for ordinary text or append mode. Supply an explicit newline; string-array elements are the exception because they are joined with newline separators." },
        BuiltinDocumentationFaq { question: "How can I write binary data?", answer: "Supply byte-valued numeric or logical data. The raw encoding makes the byte intent explicit, though the default also writes numeric bytes directly." },
        BuiltinDocumentationFaq { question: "What happens if the file cannot be opened?", answer: "A file I/O error reports the underlying cause, such as permission denial or a missing directory." },
        BuiltinDocumentationFaq { question: "Does `filewrite` create directories?", answer: "No. Create the parent directory first, for example with `mkdir`." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "fileread", target: BuiltinDocumentationLinkTarget::Builtin("fileread") },
        BuiltinDocumentationLink { label: "fwrite", target: BuiltinDocumentationLinkTarget::Builtin("fwrite") },
        BuiltinDocumentationLink { label: "fprintf", target: BuiltinDocumentationLinkTarget::Builtin("fprintf") },
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "fclose", target: BuiltinDocumentationLinkTarget::Builtin("fclose") },
        BuiltinDocumentationLink { label: "feof", target: BuiltinDocumentationLinkTarget::Builtin("feof") },
        BuiltinDocumentationLink { label: "fgets", target: BuiltinDocumentationLinkTarget::Builtin("fgets") },
        BuiltinDocumentationLink { label: "fopen", target: BuiltinDocumentationLinkTarget::Builtin("fopen") },
        BuiltinDocumentationLink { label: "fread", target: BuiltinDocumentationLinkTarget::Builtin("fread") },
        BuiltinDocumentationLink { label: "frewind", target: BuiltinDocumentationLinkTarget::Builtin("frewind") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "File write owner", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/filetext/filewrite.rs") },
            BuiltinDocumentationLink { label: "Write options", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/filetext/filewrite/options.rs") },
            BuiltinDocumentationLink { label: "Captured filesystem target", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/filetext/filewrite/target.rs") },
            BuiltinDocumentationLink { label: "Payload conversion", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/filetext/filewrite/payload.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Write content and options", location: "builtins::io::filetext::filewrite::tests::filewrite_writes_text_content" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
