use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("readWordEmbedding"),
    slug: Some("readWordEmbedding"),
    summary: "Read a word embedding model from a UTF-8 text file or zip archive.",
    description: "`readWordEmbedding(filename)` reads UTF-8 word2vec or GloVe text embedding data. Zip archives are supported when they contain an embedding text file.",
    keywords: &[
        "readWordEmbedding",
        "wordEmbedding",
        "text analytics",
        "word2vec",
        "GloVe",
    ],
    related: &[
        "word2vec",
        "vec2word",
        "trainWordEmbedding",
        "fastTextWordEmbedding",
        "doc2sequence",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported form",
            paragraphs: &[
                "emb = readWordEmbedding(filename)",
                "`emb` is a `wordEmbedding` compatibility object.",
                "The embedding text must be UTF-8.",
                "`filename` must be a scalar text path or 1-by-1 cell containing a character vector path.",
                "Every vector row must have the same dimension.",
                "A word2vec text header, when present, must match the number of vector rows in the file.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`filename` may be a string scalar, character vector, or a 1-by-1 cell array containing a character vector.",
                "Plain text files may use GloVe-style rows (`word value value ...`) or word2vec text rows with a leading `numWords dimension` header.",
                "Zip archives are scanned for a text-like embedding entry (`.txt`, `.vec`, `.glove`, or `.emb`); otherwise the first file entry is parsed as UTF-8 text.",
                "Duplicate words are collapsed so the last duplicate row supplies the vector, matching MATLAB's documented duplicate-word behavior.",
                "The returned RunMat `wordEmbedding` compatibility object exposes `Dimension` and `Vocabulary`. Vector storage is kept in an implementation-detail property used by `word2vec` and `vec2word`.",
                "Binary word2vec files, compressed formats other than zip, full pretrained model download helpers, the full fastText support-package corpus, and exact fastText optimizer parity for `trainWordEmbedding` remain tracked in the broader Text Analytics compatibility issue.",
                "Integer semantics: `readWordEmbedding` has no numeric filename role. Numeric, logical, symbolic, and provider-resident filename values are rejected before gather, provider access, or filesystem access.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &[
                "`readWordEmbedding` reads host file data and creates text/model metadata; it does not use provider kernels.",
                "`readWordEmbedding` performs host file IO and has no provider kernel.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Read An Embedding File",
            program: "emb = readWordEmbedding('vectors.vec')",
            display_output: Some("`emb` is a RunMat `wordEmbedding` compatibility object."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
                id: BuiltinExampleFixtureId {
                    local_name: "readwordembedding-1",
                },
                root: BuiltinFilesystemRoot::IsolatedWorkspace,
                entries: &[
                    BuiltinFilesystemEntry::File {
                        relative_path: "vectors.vec",
                        content: BuiltinFixtureContent::Utf8("3 2\nalpha 1 0\nbeta 0 1\ngamma 1 1\n")
                    },
                ],
            } ),
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(strcmp(class(emb), \"wordEmbedding\")); assert(emb.Dimension == 2); assert(isequal(word2vec(emb, \"alpha\"), [1 0]));",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Inspect Metadata",
            program: "emb = readWordEmbedding('vectors.vec');\nemb.Dimension\nemb.Vocabulary",
            display_output: Some("`Dimension` is the vector length, and `Vocabulary` is a string row vector of words."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
                id: BuiltinExampleFixtureId {
                    local_name: "readwordembedding-2",
                },
                root: BuiltinFilesystemRoot::IsolatedWorkspace,
                entries: &[
                    BuiltinFilesystemEntry::File {
                        relative_path: "vectors.vec",
                        content: BuiltinFixtureContent::Utf8("3 2\nalpha 1 0\nbeta 0 1\ngamma 1 1\n")
                    },
                ],
            } ),
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(emb.Dimension == 2); assert(isequal(emb.Vocabulary, [\"alpha\" \"beta\" \"gamma\"]));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does `readWordEmbedding` support zip files?",
            answer: "Yes. RunMat reads a UTF-8 embedding text file from a zip archive.",
        },
        BuiltinDocumentationFaq {
            question: "Does `readWordEmbedding` download pretrained embeddings?",
            answer: "No. This helper reads local UTF-8 text or zip files. Pretrained download helpers remain separate Text Analytics work.",
        },
        BuiltinDocumentationFaq {
            question: "Does `readWordEmbedding` execute on the GPU?",
            answer: "No. It is host file parsing and object construction.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "word2vec",
            target: BuiltinDocumentationLinkTarget::Builtin("word2vec")
        },
        BuiltinDocumentationLink {
            label: "vec2word",
            target: BuiltinDocumentationLinkTarget::Builtin("vec2word")
        },
        BuiltinDocumentationLink {
            label: "trainWordEmbedding",
            target: BuiltinDocumentationLinkTarget::Builtin("trainWordEmbedding")
        },
        BuiltinDocumentationLink {
            label: "fastTextWordEmbedding",
            target: BuiltinDocumentationLinkTarget::Builtin("fastTextWordEmbedding")
        },
        BuiltinDocumentationLink {
            label: "doc2sequence",
            target: BuiltinDocumentationLinkTarget::Builtin("doc2sequence")
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "`crates/runmat-runtime/src/builtins/strings/text_analytics/embeddings.rs`",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/text_analytics/embeddings.rs")
            },
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::strings::text_analytics::embeddings::tests",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
