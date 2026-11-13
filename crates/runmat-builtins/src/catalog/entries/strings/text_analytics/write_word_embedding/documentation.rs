use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,

    title: Some("writeWordEmbedding"),

    slug: Some("writeWordEmbedding"),

    summary: "Write a word embedding model to a UTF-8 word2vec text file.",

    description: "`writeWordEmbedding(emb, filename)` writes a `wordEmbedding` object to a UTF-8 text file in word2vec text format.",

    keywords: &["writeWordEmbedding", "wordEmbedding", "text analytics", "word2vec", "write"],
    related: &["readWordEmbedding", "trainWordEmbedding", "word2vec", "vec2word", "doc2sequence"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported form",
            paragraphs: &["writeWordEmbedding(emb, filename)", "The embedding is written to `filename`.", "`emb` must be a RunMat `wordEmbedding` object.", "`filename` must identify a writable host filesystem path as a scalar text value or 1-by-1 cell containing a character vector.", "The output text is UTF-8 and can be read back with `readWordEmbedding`."],
        },
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &["`emb` must be a `wordEmbedding` object with `Dimension`, `Vocabulary`, and vector storage created by `readWordEmbedding`, `trainWordEmbedding`, or `fastTextWordEmbedding`.", "`filename` may be a string scalar, character vector, or a 1-by-1 cell array containing a character vector.", "The output file starts with a `numWords dimension` header followed by one row per vocabulary word and its vector values.", "The target file is created or overwritten. Parent directories must already exist.", "Vocabulary entries containing whitespace and non-finite vector values are rejected because word2vec text rows cannot represent them unambiguously.", "Binary word2vec output, zip output, append mode, and format-selection options are not part of the compatibility target's documented `writeWordEmbedding` syntax and are not accepted.", "The filename must be host text. Native integer and resident numeric filename values are rejected before provider or filesystem access."],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &["`writeWordEmbedding` serializes host wordEmbedding metadata and writes a file; it does not use provider kernels.", "`writeWordEmbedding` is a host file-I/O sink and does not launch provider kernels. Resident numeric filename inputs are rejected rather than gathered because they are not valid paths."],
        },
        BuiltinDocumentationSection {
            heading: "Implementation boundary",
            paragraphs: &["The existing filesystem implementation writes UTF-8 word2vec text from host double model data. The examples write real isolated files and read them back; no trained-corpus parity or provider-resident serialization is claimed."],
        },
        BuiltinDocumentationSection {
            heading: "Finite host allocation",
            paragraphs: &["The current implementation requires an Unbounded host allocation policy. Under Managed finite allocation, a shared family guard rejects before gather, model cloning, mutation or filesystem work because complete text-model scratch and metadata ownership is not yet implemented. Authority is checked again after suspension and before publication, and failures are terminal rather than permission to replay. This refusal is an implementation frontier, not completion of finite resource support."],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Write a trained embedding",
            program: "documents = tokenizedDocument([\"red blue red\"; \"green blue\"]);\nemb = trainWordEmbedding(documents, 'Dimension', 8, 'MinCount', 1);\nwriteWordEmbedding(emb, \"colors.vec\")",
            display_output: Some("`colors.vec` contains UTF-8 word2vec text embedding data."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
                id: BuiltinExampleFixtureId {
                    local_name: "writewordembedding-1",
                },
                root: BuiltinFilesystemRoot::IsolatedWorkspace,
                entries: &[BuiltinFilesystemEntry::File {
                    relative_path: "workspace.txt",
                    content: BuiltinFixtureContent::Utf8("Isolated embedding export workspace.\n"),
                }],
            }),
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "copy = readWordEmbedding(\"colors.vec\"); assert(copy.Dimension == 8); assert(isequal(copy.Vocabulary, emb.Vocabulary)); assert(norm(word2vec(copy, copy.Vocabulary) - word2vec(emb, emb.Vocabulary), \"fro\") < 1e-10);",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Round-trip an embedding file",
            program: "documents = tokenizedDocument([\"red blue red\"; \"green blue\"]); emb = trainWordEmbedding(documents, \"Dimension\", 8, \"MinCount\", 1);\nwriteWordEmbedding(emb, \"model.vec\");\ncopy = readWordEmbedding(\"model.vec\");\ncopy.Dimension",
            display_output: Some("`copy` is a `wordEmbedding` object with the same vocabulary and vectors."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
                id: BuiltinExampleFixtureId {
                    local_name: "writewordembedding-2",
                },
                root: BuiltinFilesystemRoot::IsolatedWorkspace,
                entries: &[BuiltinFilesystemEntry::File {
                    relative_path: "workspace.txt",
                    content: BuiltinFixtureContent::Utf8("Isolated embedding export workspace.\n"),
                }],
            }),
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(copy.Dimension == emb.Dimension); assert(isequal(copy.Vocabulary, emb.Vocabulary)); assert(norm(word2vec(copy, copy.Vocabulary) - word2vec(emb, emb.Vocabulary), \"fro\") < 1e-10);",
            },
        },
    ],
    example_exemption: None,

    faqs: &[
        BuiltinDocumentationFaq {
            question: "What file format does `writeWordEmbedding` use?",
            answer: "It writes UTF-8 word2vec text format with a header row and one word-vector row per vocabulary entry.",
        },
        BuiltinDocumentationFaq {
            question: "Does `writeWordEmbedding` create folders?",
            answer: "No. Create the parent folder first with `mkdir` if needed.",
        },
        BuiltinDocumentationFaq {
            question: "Does `writeWordEmbedding` execute on the GPU?",
            answer: "No. It is host object serialization and file I/O.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "readWordEmbedding",
            target: BuiltinDocumentationLinkTarget::Builtin("readWordEmbedding"),
        },
        BuiltinDocumentationLink {
            label: "trainWordEmbedding",
            target: BuiltinDocumentationLinkTarget::Builtin("trainWordEmbedding"),
        },
        BuiltinDocumentationLink {
            label: "word2vec",
            target: BuiltinDocumentationLinkTarget::Builtin("word2vec"),
        },
        BuiltinDocumentationLink {
            label: "vec2word",
            target: BuiltinDocumentationLinkTarget::Builtin("vec2word"),
        },
        BuiltinDocumentationLink {
            label: "doc2sequence",
            target: BuiltinDocumentationLinkTarget::Builtin("doc2sequence"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/strings/text_analytics/embeddings.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/text_analytics/embeddings.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "unit",
            location: "builtins::strings::text_analytics::embeddings::tests",
        }],
        notes: &[],
    },
    introduced: None,

    status: Some(BuiltinDocumentationStatus::Partial),
};
