use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("fgets"), slug: Some("fgets"),
    summary: "Read a line while retaining its newline characters.",
    description: "`fgets(fid)` reads from the shared file cursor through the next line terminator, including the terminator. `fgets(fid, nchar)` limits the read, and a second output reports the terminator byte values. EOF before any character returns double -1.",
    keywords: &["fgets", "file", "io", "line", "newline", "nchar"],
    related: &["fopen", "fclose", "fgetl", "feof", "fread", "fileread"],
    sections: &[
        BuiltinDocumentationSection { heading: "Lines, limits, and EOF", paragraphs: &[
            "A successful read returns a host character row retaining CR, LF, CRLF, or LFCR terminators. The optional second output is a numeric row of their byte codes, empty when no terminator ended the line. At EOF before any character both outputs are double -1. A final unterminated line is still returned once.",
            "`nchar` must be a nonnegative integer-valued scalar. The current reader applies this limit to input bytes before decoding, so a limit that splits a multibyte character can raise a decode error. Later bytes stay available for the next call; zero returns an empty character row. The compatible forms use scalar double identifiers and limits. Typed integer, Single, and resident scalar variants each have separate RunMat extension gates and exact integer bounds.",
            "The identifier must be open for reading. Standard stream identifier 0 is not supported; use `input` for interactive prompts. The file's `fopen` encoding controls decoding. `fgetl` differs by removing the terminator. Check `ischar(line)` to distinguish an empty character row from numeric EOF -1.",
        ] },
        BuiltinDocumentationSection { heading: "Placement and filesystem", paragraphs: &[
            "The file handle, cursor, and returned text remain host-owned. There is no GPU kernel or fusion path. Admitted resident scalar controls are gathered from their exact provider owner after compatibility checks; the caller retains the original allocation.",
            "The installed filesystem provider supplies the file. Browser use requires an installed file service with appropriate permissions; the runnable examples use isolated fixtures rather than ambient login or a public path.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "first-line", title: "Read the first line including newline", program: "fid = fopen('lines.txt', 'r');\nassert(fid >= 3);\nline = fgets(fid);\nstatus = fclose(fid);\nline", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "first" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: &[BuiltinFilesystemEntry::File { relative_path: "lines.txt", content: BuiltinFixtureContent::Utf8("RunMat\nSecond line\n") }] }), requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(line, sprintf('RunMat\\n'))); assert(status == 0);" } },
        BuiltinExample { id: "limited", title: "Limit the number of returned characters", program: "fid = fopen('limited.txt', 'r');\nassert(fid >= 3);\nsnippet = fgets(fid, 5);\nnext = fgets(fid);\nstatus = fclose(fid);\nsnippet\nnext", display_output: Some("snippet = Examp"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "limited" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: &[BuiltinFilesystemEntry::File { relative_path: "limited.txt", content: BuiltinFixtureContent::Utf8("Example line\n") }] }), requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(snippet, 'Examp')); assert(strcmp(next, sprintf('le line\\n'))); assert(status == 0);" } },
        BuiltinExample { id: "terminator", title: "Inspect newline bytes separately", program: "fid = fopen('terminator.txt', 'r');\nassert(fid >= 3);\n[line, ltout] = fgets(fid);\nstatus = fclose(fid);\nline\nltout", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "terminator" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: &[BuiltinFilesystemEntry::File { relative_path: "terminator.txt", content: BuiltinFixtureContent::Utf8("Line\n") }] }), requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(line, sprintf('Line\\n'))); assert(numel(ltout) == 1 && ltout(1) == 10); assert(status == 0);" } },
        BuiltinExample { id: "unterminated", title: "Read the final line without newline", program: "fid = fopen('final.txt', 'r');\nassert(fid >= 3);\nfirst = fgets(fid);\nsecond = fgets(fid);\nstatus = fclose(fid);\nfirst\nsecond", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "final" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: &[BuiltinFilesystemEntry::File { relative_path: "final.txt", content: BuiltinFixtureContent::Utf8("last line") }] }), requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(first, 'last line')); assert(second == -1); assert(status == 0);" } },
        BuiltinExample { id: "eof-loop",
            title: "Read lines until the numeric EOF sentinel",
            program: "fid = fopen('two.txt', 'r');\nassert(fid >= 3);\ncount = 0;\nline = fgets(fid);\nwhile ischar(line)\n    count = count + 1;\n    if count == 1\n        assert(strcmp(line, sprintf('one\\n')));\n    else\n        assert(count == 2);\n        assert(strcmp(line, sprintf('two\\n')));\n    end\n    fprintf('> %s', line);\n    line = fgets(fid);\nend\nstatus = fclose(fid);",
            display_output: Some("> one\n> two\n"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "two" },
            root: BuiltinFilesystemRoot::IsolatedWorkspace,
            entries: &[BuiltinFilesystemEntry::File { relative_path: "two.txt",
            content: BuiltinFixtureContent::Utf8("one\ntwo\n") }] }),
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(count == 2); assert(line == -1); assert(status == 0);" } },
        BuiltinExample { id: "latin1", title: "Decode text with the recorded Latin-1 encoding", program: "fid = fopen('latin1.txt', 'r', 'n', 'latin1');\nassert(fid >= 3);\nline = fgets(fid);\nstatus = fclose(fid);\nline", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "latin1" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: &[BuiltinFilesystemEntry::File { relative_path: "latin1.txt", content: BuiltinFixtureContent::Bytes(&[69, 115, 112, 97, 241, 111, 108, 10]) }] }), requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(line, sprintf('Español\\n'))); assert(status == 0);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What is returned at EOF?", answer: "Double -1 when no character can be read; successful lines are character rows, including an empty row for a zero-character limit." },
        BuiltinDocumentationFaq { question: "Are newline characters stripped?", answer: "No. `fgets` retains them; `fgetl` removes them." },
        BuiltinDocumentationFaq { question: "What is the second output?", answer: "A numeric row of terminating byte codes, empty if the read stopped without a terminator, or double -1 at EOF before any character." },
        BuiltinDocumentationFaq { question: "What does `nchar` do?", answer: "The current reader limits input bytes before text decoding; content beyond the limit remains unread, and splitting a multibyte character can fail decoding." },
        BuiltinDocumentationFaq { question: "Which encoding applies?", answer: "The encoding recorded by `fopen` is used to decode the line." },
        BuiltinDocumentationFaq { question: "Can it read a write-only file or stdin?", answer: "No. The handle must have read permission; interactive standard input identifier 0 is not supported here." },
        BuiltinDocumentationFaq { question: "How do I loop reliably?", answer: "Read while `ischar(line)` and close the identifier afterward; EOF is the numeric -1 sentinel." },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "fopen",
            target: BuiltinDocumentationLinkTarget::Builtin("fopen"),
        },
        BuiltinDocumentationLink {
            label: "fclose",
            target: BuiltinDocumentationLinkTarget::Builtin("fclose"),
        },
        BuiltinDocumentationLink {
            label: "fgetl",
            target: BuiltinDocumentationLinkTarget::Builtin("fgetl"),
        },
        BuiltinDocumentationLink {
            label: "feof",
            target: BuiltinDocumentationLinkTarget::Builtin("feof"),
        },
        BuiltinDocumentationLink {
            label: "fread",
            target: BuiltinDocumentationLinkTarget::Builtin("fread"),
        },
        BuiltinDocumentationLink {
            label: "fileread",
            target: BuiltinDocumentationLinkTarget::Builtin("fileread"),
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
        implementation: &[BuiltinDocumentationLink { label: "Line reader", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/filetext/fgets.rs") }, BuiltinDocumentationLink { label: "Shared cursor owner", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/filetext/registry.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Line and terminator semantics", location: "builtins::io::filetext::fgets::tests::fgets_reads_line_with_newline" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
