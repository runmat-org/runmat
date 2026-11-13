use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("extractFileText"),
    slug: Some("extractFileText"),
    summary: "Read text from plain text, HTML, and DOCX files.",
    description: "`extractFileText(filename)` reads text from a host file and returns a string scalar. This slice supports plain text files with encodings, HTML files through RunMat's `extractHTMLText` parser, and DOCX files by extracting `word/document.xml` text.",
    keywords: &["extractFileText", "text analytics", "file", "HTML", "DOCX", "plain text"],
    related: &["extractHTMLText", "tokenizedDocument", "fileread"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported forms",
            paragraphs: &["str = extractFileText(filename)\nstr = extractFileText(filename, \"Encoding\", encoding)\nstr = extractFileText(filename, \"ExtractionMethod\", method)", "`str` is a string scalar.", "MATLAB documents `Encoding` for plain text. RunMat currently also applies it while decoding HTML input.", "`ExtractionMethod` is supported for HTML files only.", "`Password` and `Pages` are recognized as PDF-only options. Pages values are validated exactly, but extraction remains unsupported until PDF support is implemented."],
        },
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "Documented `filename`/URL input is a string scalar, character vector, or 1-by-1 cell containing a character vector. A string-valued or nested filename cell is a RunMat extension.",
                "Plain text files are decoded as UTF-8 by default. The `Encoding` name-value option accepts labels supported by `encoding_rs`, such as `UTF-8`, `windows-1252`, and `ISO-8859-1`.",
                "HTML files with extensions `.html`, `.htm`, and `.xhtml` are decoded as text and passed to `extractHTMLText`.",
                "For HTML files, `ExtractionMethod` accepts `\"tree\"`, `\"article\"`, or `\"all-text\"` and follows the existing `extractHTMLText` compatibility behavior.",
                "DOCX files are read as zip archives and text is extracted from `word/document.xml`, preserving paragraph breaks and basic tabs/line breaks.",
                "The documented `Pages` option is a nonempty vector of positive values in single, double, or any of the eight integer classes. RunMat parses these controls exactly from native storage, but PDF extraction itself remains unsupported.",
                "PDF extraction, website/URL fetching, password-protected documents, legacy binary DOC, and full Microsoft Word layout fidelity remain tracked by the broader Text Analytics compatibility issue.",
                "Integer, logical, complex, and resident numeric filename values reject before file IO or provider access. Resident Pages controls require RunMat extension mode.",
                "Unknown file extensions are treated as plain text files.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &["`extractFileText` performs host file IO and text parsing. It has no provider kernel; resident Pages controls are available only through an explicit RunMat extension.", "`extractFileText` performs host file IO and text parsing and has no provider kernel."],
        },
        BuiltinDocumentationSection {
            heading: "Implementation boundary",
            paragraphs: &["Plain text and HTML use the existing filesystem/encoding owners; DOCX reads a real ZIP main-document part. PDF, URL fetching, protected documents and full layout fidelity remain unsupported. The file-size check follows read completion and is not a bounded streaming or finite-allocation claim."],
        },
        BuiltinDocumentationSection {
            heading: "Finite host allocation",
            paragraphs: &["The current implementation requires an Unbounded host allocation policy. Under Managed finite allocation, a shared family guard rejects before gather, parsing, model cloning, chart mutation or filesystem work because complete host scratch and metadata ownership is not yet implemented. Authority is checked again after suspension and before publication, and failures are terminal rather than permission to replay. This refusal is an implementation frontier, not completion of finite resource support."],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Read Plain Text",
            program: "str = extractFileText(\"notes.txt\")",
            display_output: Some("`str` is the file contents as a string scalar."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
                id: BuiltinExampleFixtureId {
                    local_name: "extractfiletext-1",
                },
                root: BuiltinFilesystemRoot::IsolatedWorkspace,
                entries: &[BuiltinFilesystemEntry::File {
                    relative_path: "notes.txt",
                    content: BuiltinFixtureContent::Utf8("Alpha notes\nBeta notes\n"),
                }],
            }),
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(strcmp(str, sprintf('Alpha notes\\nBeta notes\\n')));",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Read HTML Text",
            program: "str = extractFileText(\"page.html\", \"ExtractionMethod\", \"all-text\")",
            display_output: Some("`str` contains visible text extracted from the HTML file."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
                id: BuiltinExampleFixtureId {
                    local_name: "extractfiletext-2",
                },
                root: BuiltinFilesystemRoot::IsolatedWorkspace,
                entries: &[BuiltinFilesystemEntry::File {
                    relative_path: "page.html",
                    content: BuiltinFixtureContent::Utf8("<html><head><title>Hidden title</title></head><body><h1>Visible title</h1><p>Visible &amp; useful</p><script>hidden()</script></body></html>"),
                }],
            }),
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(strcmp(str, \"Visible title Visible & useful\"));",
            },
        },
        BuiltinExample {
            id: "retained-3",
            title: "Read DOCX Text",
            program: "str = extractFileText(\"report.docx\")",
            display_output: Some("`str` contains the text runs from the DOCX main document part."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
                id: BuiltinExampleFixtureId {
                    local_name: "extractfiletext-3",
                },
                root: BuiltinFilesystemRoot::IsolatedWorkspace,
                entries: &[BuiltinFilesystemEntry::File {
                    relative_path: "report.docx",
                    content: BuiltinFixtureContent::Bytes(&[
                        80, 75, 3, 4, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 186, 119, 164, 156, 83, 1, 0, 0, 83, 1, 0, 0, 19, 0, 0, 0, 91, 67, 111, 110, 116, 101, 110, 116, 95, 84, 121, 112, 101, 115, 93, 46, 120, 109, 108, 60, 63, 120, 109, 108, 32, 118, 101, 114, 115, 105, 111, 110, 61, 34, 49, 46, 48, 34, 63, 62, 60, 84, 121, 112, 101, 115, 32, 120, 109, 108, 110, 115, 61, 34, 104, 116, 116, 112, 58, 47, 47, 115, 99, 104, 101, 109, 97, 115, 46, 111, 112, 101, 110, 120, 109, 108, 102, 111, 114, 109, 97, 116, 115, 46, 111, 114, 103, 47, 112, 97, 99, 107, 97, 103, 101, 47, 50, 48, 48, 54, 47, 99, 111, 110, 116, 101, 110, 116, 45, 116, 121, 112, 101, 115, 34, 62, 60, 68, 101, 102, 97, 117, 108, 116, 32, 69, 120, 116, 101, 110, 115, 105, 111, 110, 61, 34, 114, 101, 108, 115, 34, 32, 67, 111, 110, 116, 101, 110, 116, 84, 121, 112, 101, 61, 34, 97, 112, 112, 108, 105, 99, 97, 116, 105, 111, 110, 47, 118, 110, 100, 46, 111, 112, 101, 110, 120, 109, 108, 102, 111, 114, 109, 97, 116, 115, 45, 112,
                        97, 99, 107, 97, 103, 101, 46, 114, 101, 108, 97, 116, 105, 111, 110, 115, 104, 105, 112, 115, 43, 120, 109, 108, 34, 47, 62, 60, 79, 118, 101, 114, 114, 105, 100, 101, 32, 80, 97, 114, 116, 78, 97, 109, 101, 61, 34, 47, 119, 111, 114, 100, 47, 100, 111, 99, 117, 109, 101, 110, 116, 46, 120, 109, 108, 34, 32, 67, 111, 110, 116, 101, 110, 116, 84, 121, 112, 101, 61, 34, 97, 112, 112, 108, 105, 99, 97, 116, 105, 111, 110, 47, 118, 110, 100, 46, 111, 112, 101, 110, 120, 109, 108, 102, 111, 114, 109, 97, 116, 115, 45, 111, 102, 102, 105, 99, 101, 100, 111, 99, 117, 109, 101, 110, 116, 46, 119, 111, 114, 100, 112, 114, 111, 99, 101, 115, 115, 105, 110, 103, 109, 108, 46, 100, 111, 99, 117, 109, 101, 110, 116, 46, 109, 97, 105, 110, 43, 120, 109, 108, 34, 47, 62, 60, 47, 84, 121, 112, 101, 115, 62, 80, 75, 3, 4, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 95, 51, 149, 82, 7, 1, 0, 0, 7, 1, 0, 0, 11, 0, 0, 0, 95, 114, 101, 108, 115, 47, 46, 114, 101, 108, 115, 60, 63, 120, 109,
                        108, 32, 118, 101, 114, 115, 105, 111, 110, 61, 34, 49, 46, 48, 34, 63, 62, 60, 82, 101, 108, 97, 116, 105, 111, 110, 115, 104, 105, 112, 115, 32, 120, 109, 108, 110, 115, 61, 34, 104, 116, 116, 112, 58, 47, 47, 115, 99, 104, 101, 109, 97, 115, 46, 111, 112, 101, 110, 120, 109, 108, 102, 111, 114, 109, 97, 116, 115, 46, 111, 114, 103, 47, 112, 97, 99, 107, 97, 103, 101, 47, 50, 48, 48, 54, 47, 114, 101, 108, 97, 116, 105, 111, 110, 115, 104, 105, 112, 115, 34, 62, 60, 82, 101, 108, 97, 116, 105, 111, 110, 115, 104, 105, 112, 32, 73, 100, 61, 34, 114, 73, 100, 49, 34, 32, 84, 121, 112, 101, 61, 34, 104, 116, 116, 112, 58, 47, 47, 115, 99, 104, 101, 109, 97, 115, 46, 111, 112, 101, 110, 120, 109, 108, 102, 111, 114, 109, 97, 116, 115, 46, 111, 114, 103, 47, 111, 102, 102, 105, 99, 101, 68, 111, 99, 117, 109, 101, 110, 116, 47, 50, 48, 48, 54, 47, 114, 101, 108, 97, 116, 105, 111, 110, 115, 104, 105, 112, 115, 47, 111, 102, 102, 105, 99, 101, 68, 111, 99, 117,
                        109, 101, 110, 116, 34, 32, 84, 97, 114, 103, 101, 116, 61, 34, 119, 111, 114, 100, 47, 100, 111, 99, 117, 109, 101, 110, 116, 46, 120, 109, 108, 34, 47, 62, 60, 47, 82, 101, 108, 97, 116, 105, 111, 110, 115, 104, 105, 112, 115, 62, 80, 75, 3, 4, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 223, 36, 12, 173, 236, 0, 0, 0, 236, 0, 0, 0, 17, 0, 0, 0, 119, 111, 114, 100, 47, 100, 111, 99, 117, 109, 101, 110, 116, 46, 120, 109, 108, 60, 63, 120, 109, 108, 32, 118, 101, 114, 115, 105, 111, 110, 61, 34, 49, 46, 48, 34, 63, 62, 60, 119, 58, 100, 111, 99, 117, 109, 101, 110, 116, 32, 120, 109, 108, 110, 115, 58, 119, 61, 34, 104, 116, 116, 112, 58, 47, 47, 115, 99, 104, 101, 109, 97, 115, 46, 111, 112, 101, 110, 120, 109, 108, 102, 111, 114, 109, 97, 116, 115, 46, 111, 114, 103, 47, 119, 111, 114, 100, 112, 114, 111, 99, 101, 115, 115, 105, 110, 103, 109, 108, 47, 50, 48, 48, 54, 47, 109, 97, 105, 110, 34, 62, 60, 119, 58, 98, 111, 100, 121, 62, 60, 119, 58, 112, 62, 60, 119, 58,
                        114, 62, 60, 119, 58, 116, 62, 81, 117, 97, 114, 116, 101, 114, 108, 121, 32, 114, 101, 112, 111, 114, 116, 60, 47, 119, 58, 116, 62, 60, 47, 119, 58, 114, 62, 60, 47, 119, 58, 112, 62, 60, 119, 58, 112, 62, 60, 119, 58, 114, 62, 60, 119, 58, 116, 62, 82, 101, 118, 101, 110, 117, 101, 32, 38, 97, 109, 112, 59, 32, 103, 114, 111, 119, 116, 104, 60, 47, 119, 58, 116, 62, 60, 47, 119, 58, 114, 62, 60, 47, 119, 58, 112, 62, 60, 47, 119, 58, 98, 111, 100, 121, 62, 60, 47, 119, 58, 100, 111, 99, 117, 109, 101, 110, 116, 62, 80, 75, 1, 2, 20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 186, 119, 164, 156, 83, 1, 0, 0, 83, 1, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 91, 67, 111, 110, 116, 101, 110, 116, 95, 84, 121, 112, 101, 115, 93, 46, 120, 109, 108, 80, 75, 1, 2, 20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 95, 51, 149, 82, 7, 1, 0, 0, 7, 1, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 132, 1, 0, 0, 95, 114, 101, 108, 115, 47, 46, 114, 101, 108, 115, 80, 75,
                        1, 2, 20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 223, 36, 12, 173, 236, 0, 0, 0, 236, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 180, 2, 0, 0, 119, 111, 114, 100, 47, 100, 111, 99, 117, 109, 101, 110, 116, 46, 120, 109, 108, 80, 75, 5, 6, 0, 0, 0, 0, 3, 0, 3, 0, 185, 0, 0, 0, 207, 3, 0, 0, 0, 0,
                    ]),
                }],
            }),
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(strcmp(str, sprintf('Quarterly report\\nRevenue & growth')));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does `extractFileText` support PDF files?",
            answer: "Not yet. RunMat rejects PDF-specific forms with a clear error and keeps PDF extraction in the broader Text Analytics queue.",
        },
        BuiltinDocumentationFaq {
            question: "Does `extractFileText` read websites?",
            answer: "Not in this slice. URL extraction requires web fetching behavior and remains tracked.",
        },
        BuiltinDocumentationFaq {
            question: "Does `extractFileText` execute on the GPU?",
            answer: "No. It performs host file IO and text parsing.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "extractHTMLText",
            target: BuiltinDocumentationLinkTarget::Builtin("extractHTMLText"),
        },
        BuiltinDocumentationLink {
            label: "tokenizedDocument",
            target: BuiltinDocumentationLinkTarget::Builtin("tokenizedDocument"),
        },
        BuiltinDocumentationLink {
            label: "fileread",
            target: BuiltinDocumentationLinkTarget::Builtin("fileread"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/strings/text_analytics/file_text.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/text_analytics/file_text.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "unit",
            location: "builtins::strings::text_analytics::file_text::tests",
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
