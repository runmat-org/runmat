use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("fgetl"), slug: Some("fgetl"),
    summary: "Read a text line without its newline characters.",
    description: "`fgetl(fid)` reads the next line from an open readable file, removes its trailing line terminator, and returns a character row. A read attempted at end-of-file returns double -1; a blank line remains an empty character row.",
    keywords: &["fgetl", "file", "io", "line", "newline"],
    related: &["fopen", "fclose", "feof", "fgets", "fread", "fileread"],
    sections: &[
        BuiltinDocumentationSection { heading: "Reading and EOF", paragraphs: &[
            "A successful read advances the shared file cursor. CR, LF, and paired CRLF or LFCR terminators are removed from the returned row; a final line without a terminator remains readable. An empty line is an empty character row, not the numeric -1 sentinel.",
            "The compatible input is a finite integer-valued double scalar identifier from `fopen`. Typed integer, Single, and provider-resident numeric scalar identifiers have separate RunMat extension gates. Standard input/output identifiers are not supported for `fgetl`; a closed, unreadable, or invalid file identifier errors.",
            "At EOF, a read attempt returns double -1 and sets the registry EOF flag queried by `feof`; `frewind` clears the flag. Extra requested outputs follow the runtime's legacy double-zero padding. `fgets` differs by retaining the line terminator and optionally reporting its byte values.",
        ] },
        BuiltinDocumentationSection { heading: "Placement and filesystem", paragraphs: &[
            "The file handle, cursor, and returned text are host-owned. There is no GPU kernel or fusion path. An admitted resident scalar identifier is gathered before host registry lookup; the original provider allocation remains owned by its caller.",
            "The installed filesystem service must support the open file. Browser filesystem access depends on an installed browser file service and its permissions; the examples use isolated fixture files rather than an ambient browser login.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "first-line", title: "Read the first line without a newline", program: "fid = fopen('lines.txt', 'r');\nassert(fid >= 3);\nline = fgetl(fid);\nstatus = fclose(fid);\nline", display_output: Some("line = RunMat"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "first" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: &[BuiltinFilesystemEntry::File { relative_path: "lines.txt", content: BuiltinFixtureContent::Utf8("RunMat\nSecond line\n") }] }), requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(line, 'RunMat')); assert(status == 0);" } },
        BuiltinExample { id: "line-loop", title: "Read lines until the EOF sentinel", program: "fid = fopen('twolines.txt', 'r');\nassert(fid >= 3);\ncount = 0;\nline = fgetl(fid);\nwhile ischar(line)\n    count = count + 1;\n    line = fgetl(fid);\nend\nstatus = fclose(fid);\ncount", display_output: Some("count = 2"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "loop" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: &[BuiltinFilesystemEntry::File { relative_path: "twolines.txt", content: BuiltinFixtureContent::Utf8("one\ntwo\n") }] }), requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(count == 2); assert(line == -1); assert(status == 0);" } },
        BuiltinExample { id: "empty-line", title: "Distinguish an empty line from EOF", program: "fid = fopen('emptyline.txt', 'r');\nassert(fid >= 3);\nfirst = fgetl(fid);\nsecond = fgetl(fid);\nthird = fgetl(fid);\nstatus = fclose(fid);\nfirst\nsecond\nthird", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "empty-line" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: &[BuiltinFilesystemEntry::File { relative_path: "emptyline.txt", content: BuiltinFixtureContent::Utf8("\nlast") }] }), requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(ischar(first) && isempty(first)); assert(strcmp(second, 'last')); assert(third == -1); assert(status == 0);" } },
        BuiltinExample { id: "unterminated-final", title: "Read a final line without a newline", program: "fid = fopen('final.txt', 'r');\nassert(fid >= 3);\nfirst = fgetl(fid);\nsecond = fgetl(fid);\nstatus = fclose(fid);\nfirst\nsecond", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "unterminated" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: &[BuiltinFilesystemEntry::File { relative_path: "final.txt", content: BuiltinFixtureContent::Utf8("last line") }] }), requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(first, 'last line')); assert(second == -1); assert(status == 0);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What does `fgetl` return at EOF?", answer: "Double -1 after a read attempt at the end; an empty line is an empty character row." },
        BuiltinDocumentationFaq { question: "How does it differ from `fgets`?", answer: "`fgetl` strips the line terminator, while `fgets` retains it and can return terminator byte values." },
        BuiltinDocumentationFaq { question: "What happens for a blank line?", answer: "It returns an empty character row and advances past that line; this is distinct from numeric -1." },
        BuiltinDocumentationFaq { question: "Can it read from a write-only file?", answer: "No. The identifier must refer to a readable open file." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "fopen", target: BuiltinDocumentationLinkTarget::Builtin("fopen") },
        BuiltinDocumentationLink { label: "fclose", target: BuiltinDocumentationLinkTarget::Builtin("fclose") },
        BuiltinDocumentationLink { label: "feof", target: BuiltinDocumentationLinkTarget::Builtin("feof") },
        BuiltinDocumentationLink { label: "fgets", target: BuiltinDocumentationLinkTarget::Builtin("fgets") },
        BuiltinDocumentationLink { label: "fread", target: BuiltinDocumentationLinkTarget::Builtin("fread") },
        BuiltinDocumentationLink { label: "fileread", target: BuiltinDocumentationLinkTarget::Builtin("fileread") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Line read owner", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/filetext/fgetl.rs") }, BuiltinDocumentationLink { label: "Shared file registry", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/filetext/registry.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Line and EOF cases", location: "builtins::io::filetext::fgetl::tests::fgetl_reads_line_without_newline" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
