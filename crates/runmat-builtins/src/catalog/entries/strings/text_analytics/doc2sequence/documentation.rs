use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,

    title: Some("doc2sequence"),

    slug: Some("doc2sequence"),

    summary: "Convert tokenized documents to word-vector or word-index sequences.",

    description: "`doc2sequence(emb, documents)` returns a cell array whose i-th cell contains the embedding-vector sequence for the i-th tokenized document. `doc2sequence(enc, documents)` returns word-index sequences from a `wordEncoding` object.",

    keywords: &["doc2sequence", "wordEmbedding", "wordEncoding", "tokenizedDocument", "text analytics", "sequence"],
    related: &["fastTextWordEmbedding", "word2vec", "wordEncoding", "vec2word", "tokenizedDocument", "trainWordEmbedding"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported forms",
            paragraphs: &["sequences = doc2sequence(emb, documents)\nsequences = doc2sequence(enc, documents)\nsequences = doc2sequence(___, Name, Value)", "`sequences` is a cell array of embedding-vector matrices or word-index vectors.", "`UnknownWord` must be `\"discard\"` or `\"nan\"`.", "`PaddingDirection` must be `\"left\"`, `\"right\"`, or `\"none\"`.", "`PaddingValue` must be a numeric scalar.", "`Length` must be `\"longest\"`, `\"shortest\"`, or a positive integer."],
        },
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "The first input must be a RunMat `wordEmbedding` or `wordEncoding` compatibility object.",
                "`documents` must be a RunMat `tokenizedDocument` compatibility object.",
                "For embedding input, each output cell contains a D-by-S numeric matrix, where D is the embedding dimension and S is the sequence length.",
                "For encoding input, each output cell contains a 1-by-S numeric vector of 1-based word encoding indices.",
                "Unknown words are discarded by default. Set `UnknownWord` to `\"nan\"` to include NaN vector columns for embedding inputs or NaN index elements for encoding inputs.",
                "`PaddingDirection` accepts `\"left\"`, `\"right\"`, and `\"none\"`. The default is left padding.",
                "`PaddingValue` accepts single, double, or any of the eight integer scalar classes and defaults to 0. Typed storage crosses explicitly to the floating output class. MATLAB warns against NaN padding for deep learning workflows, but RunMat accepts numeric NaN for compatibility.",
                "`Length` accepts `\"longest\"`, `\"shortest\"`, or a positive integer in single, double, or any of the eight integer classes. Typed integer scalars are validated exactly; positive fixed lengths truncate sequences on the right.",
                "When `PaddingDirection` is `\"none\"`, output cells keep per-document sequence lengths, except that fixed or shortest `Length` values still truncate longer documents.",
                "Very large padded dense outputs are rejected with a controlled error to avoid accidental memory explosions.",
                "Resident tensors are rejected recursively before provider access, including tensors nested inside compatibility objects such as the `__Vectors` storage of a `wordEmbedding`.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &["`doc2sequence` consumes host compatibility objects and returns host cell/tensor outputs. No GPU-array capability is documented; top-level and recursively nested resident arguments are rejected before gather/provider access.", "`doc2sequence` performs host text/object sequence construction and has no provider kernel. Top-level and recursively nested resident arguments are rejected before any gather or provider access."],
        },
        BuiltinDocumentationSection {
            heading: "Implementation boundary",
            paragraphs: &["Sequences are dense host double matrices in host cells, with the existing 50,000,000-value guard. The compact fastText model does not contain alpha or beta: the first example deliberately demonstrates default unknown-word discard and produces 300-by-0 matrices. Later examples use explicit small embedding files to exercise nonempty padding and lookup; full pretrained-corpus parity is not implied."],
        },
        BuiltinDocumentationSection {
            heading: "Finite host allocation",
            paragraphs: &["The current implementation requires an Unbounded host allocation policy. Under Managed finite allocation, a shared family guard rejects before gather, model cloning, mutation or filesystem work because complete text-model scratch and metadata ownership is not yet implemented. Authority is checked again after suspension and before publication, and failures are terminal rather than permission to replay. This refusal is an implementation frontier, not completion of finite resource support."],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Convert Documents To Word Vectors",
            program: "emb = fastTextWordEmbedding;\ndocuments = tokenizedDocument([\"alpha beta\"; \"beta\"]);\nsequences = doc2sequence(emb, documents)",
            display_output: Some("`sequences` is a cell array of D-by-S matrices."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(iscell(sequences)); assert(isequal(size(sequences), [2 1])); assert(isequal(size(sequences{1}), [300 0])); assert(isequal(size(sequences{2}), [300 0]));",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Convert Documents To Word Indices",
            program: "documents = tokenizedDocument([\"alpha beta\"; \"beta\"]);\nenc = wordEncoding(documents);\nsequences = doc2sequence(enc, documents)",
            display_output: Some("`sequences` is a cell array of 1-by-S index vectors."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(iscell(sequences)); assert(isequal(sequences{1}, [1 2])); assert(isequal(sequences{2}, [0 2]));",
            },
        },
        BuiltinExample {
            id: "retained-3",
            title: "Avoid Padding",
            program: "emb = readWordEmbedding(\"tokens.vec\"); documents = tokenizedDocument([\"alpha beta\"; \"beta\"]);\nsequences = doc2sequence(emb, documents, \"PaddingDirection\", \"none\")",
            display_output: Some("Each cell keeps the sequence length of the corresponding document."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
                id: BuiltinExampleFixtureId {
                    local_name: "doc2sequence-3",
                },
                root: BuiltinFilesystemRoot::IsolatedWorkspace,
                entries: &[BuiltinFilesystemEntry::File {
                    relative_path: "tokens.vec",
                    content: BuiltinFixtureContent::Utf8("2 2\nalpha 1 0\nbeta 0 1\n"),
                }],
            }),
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(sequences{1}, [1 0; 0 1])); assert(isequal(sequences{2}, [0; 1]));",
            },
        },
        BuiltinExample {
            id: "retained-4",
            title: "Control Padding And Unknown Words",
            program: "emb = readWordEmbedding(\"tokens.vec\"); documents = tokenizedDocument([\"alpha beta\"; \"beta\"]);\nsequences = doc2sequence(emb, documents, \"UnknownWord\", \"nan\", \"PaddingDirection\", \"right\", \"PaddingValue\", -1, \"Length\", 100)",
            display_output: Some("Sequences are right-padded or right-truncated to length 100."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
                id: BuiltinExampleFixtureId {
                    local_name: "doc2sequence-4",
                },
                root: BuiltinFilesystemRoot::IsolatedWorkspace,
                entries: &[BuiltinFilesystemEntry::File {
                    relative_path: "tokens.vec",
                    content: BuiltinFixtureContent::Utf8("2 2\nalpha 1 0\nbeta 0 1\n"),
                }],
            }),
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(sequences{1}), [2 100])); assert(isequal(sequences{1}(:,1:2), [1 0; 0 1])); assert(all(all(sequences{1}(:,3:100) == -1))); assert(isequal(sequences{2}(:,1), [0; 1]));",
            },
        },
    ],
    example_exemption: None,

    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does RunMat support `doc2sequence(enc, documents)` for `wordEncoding`?",
            answer: "Yes. RunMat returns one 1-by-S numeric index vector per tokenized document.",
        },
        BuiltinDocumentationFaq {
            question: "Does `doc2sequence` return single precision like MATLAB examples?",
            answer: "RunMat numeric tensors currently use host f64 storage, so the compatibility output is numeric but not MATLAB's exact single-precision storage.",
        },
        BuiltinDocumentationFaq {
            question: "Does `doc2sequence` execute on the GPU?",
            answer: "No. The current public documentation does not list a GPU-array capability. RunMat recursively rejects top-level or object-nested resident inputs before provider access and maps host text/object metadata to host cell and tensor outputs.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "fastTextWordEmbedding",
            target: BuiltinDocumentationLinkTarget::Builtin("fastTextWordEmbedding"),
        },
        BuiltinDocumentationLink {
            label: "word2vec",
            target: BuiltinDocumentationLinkTarget::Builtin("word2vec"),
        },
        BuiltinDocumentationLink {
            label: "wordEncoding",
            target: BuiltinDocumentationLinkTarget::Builtin("wordEncoding"),
        },
        BuiltinDocumentationLink {
            label: "vec2word",
            target: BuiltinDocumentationLinkTarget::Builtin("vec2word"),
        },
        BuiltinDocumentationLink {
            label: "tokenizedDocument",
            target: BuiltinDocumentationLinkTarget::Builtin("tokenizedDocument"),
        },
        BuiltinDocumentationLink {
            label: "trainWordEmbedding",
            target: BuiltinDocumentationLinkTarget::Builtin("trainWordEmbedding"),
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
