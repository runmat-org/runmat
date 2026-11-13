use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("trainWordEmbedding"),
    slug: Some("trainWordEmbedding"),
    summary: "Train a word embedding from a text file or tokenized documents.",
    description: "`trainWordEmbedding(filename)` and `trainWordEmbedding(documents)` return a `wordEmbedding` trained from local text. File input is UTF-8 text with one document per line and whitespace-separated words; `tokenizedDocument` input uses its stored tokens.",
    keywords: &[
        "trainWordEmbedding",
        "wordEmbedding",
        "text analytics",
        "training",
        "embedding",
    ],
    related: &[
        "readWordEmbedding",
        "word2vec",
        "vec2word",
        "tokenizedDocument",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported forms",
            paragraphs: &[
                "emb = trainWordEmbedding(filename)\nemb = trainWordEmbedding(documents)\nemb = trainWordEmbedding(___, Name, Value)",
                "`emb` is a `wordEmbedding` object.",
                "`Dimension`, `NumEpochs`, `MinCount`, `NumNegativeSamples`, and `UpdateRate` must be positive integers.",
                "`Window` must be a nonnegative integer.",
                "`Model` must be `\"skipgram\"` or `\"cbow\"`.",
                "`LossFunction` must be `\"ns\"`, `\"hs\"`, or `\"softmax\"`; `NumNegativeSamples` is only valid with `\"ns\"`.",
                "`NGramRange` must be a two-element nonnegative integer vector `[min max]` with `min <= max`; `[0 0]` disables subword features.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`filename` may be a string scalar, character vector, scalar string array, or 1-by-1 cell containing a text filename.",
                "`documents` must be a `tokenizedDocument` object.",
                "`Dimension`, `Window`, `Model`, `DiscardFactor`, `LossFunction`, `NumNegativeSamples`, `NumEpochs`, `MinCount`, `NGramRange`, `InitialLearnRate`, `UpdateRate`, and `Verbose` are parsed and validated using the documented MATLAB option names.",
                "`MinCount` filters vocabulary words before model construction. Vocabulary order is descending frequency with first-seen order as the tie-breaker.",
                "The returned object exposes `Dimension` and `Vocabulary` and stores dense vectors in the same implementation-detail property used by `word2vec` and `vec2word`.",
                "The current trainer is deterministic and uses lexical/subword hashing with windowed co-occurrence features. It supports repeatable embedding lookup workflows but does not implement the fastText optimizer.",
                "RunMat mode accepts native integer option values. Counts, sizes, ranges, and `Verbose` are decoded exactly; integer-valued learning-rate controls must be exactly representable as double.",
                "Very large dense embedding outputs are rejected before allocation when they exceed the configured model-size limit.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &[
                "`trainWordEmbedding` tokenizes host file or document data and builds host model metadata; it does not use provider kernels.",
                "`trainWordEmbedding` performs host text processing and model construction and has no provider kernel.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Train From File",
            program: "emb = trainWordEmbedding(\"training.txt\", \"Dimension\", 50, \"MinCount\", 1)",
            display_output: Some("`emb` is a `wordEmbedding` object."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
                id: BuiltinExampleFixtureId {
                    local_name: "trainwordembedding-1",
                },
                root: BuiltinFilesystemRoot::IsolatedWorkspace,
                entries: &[
                    BuiltinFilesystemEntry::File {
                        relative_path: "training.txt",
                        content: BuiltinFixtureContent::Utf8("alpha beta alpha\nbeta gamma alpha\n")
                    },
                ],
            } ),
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(strcmp(class(emb), \"wordEmbedding\")); assert(emb.Dimension == 50); assert(isequal(emb.Vocabulary, [\"alpha\" \"beta\" \"gamma\"])); assert(isequal(size(word2vec(emb, \"alpha\")), [1 50]));",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Train From Tokenized Documents",
            program: "documents = tokenizedDocument([\"alpha beta\"; \"beta gamma\"]);\nemb = trainWordEmbedding(documents, \"Window\", 2, \"MinCount\", 1)",
            display_output: Some("`emb.Vocabulary` contains the retained document tokens."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(emb.Vocabulary, [\"beta\" \"alpha\" \"gamma\"])); assert(emb.Dimension == 100);",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does RunMat implement MATLAB's exact fastText training algorithm?",
            answer: "No. The current deterministic lexical/subword and co-occurrence trainer produces a usable `wordEmbedding`, but its vectors are not fastText optimizer results.",
        },
        BuiltinDocumentationFaq {
            question: "Can the trained model be used with `word2vec` and `vec2word`?",
            answer: "Yes. The output is the `wordEmbedding` object consumed by the existing embedding lookup builtins.",
        },
        BuiltinDocumentationFaq {
            question: "Does `trainWordEmbedding` execute on the GPU?",
            answer: "No. It reads host text/model data and builds host vectors.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "readWordEmbedding",
            target: BuiltinDocumentationLinkTarget::Builtin("readWordEmbedding")
        },
        BuiltinDocumentationLink {
            label: "word2vec",
            target: BuiltinDocumentationLinkTarget::Builtin("word2vec")
        },
        BuiltinDocumentationLink {
            label: "vec2word",
            target: BuiltinDocumentationLinkTarget::Builtin("vec2word")
        },
        BuiltinDocumentationLink {
            label: "tokenizedDocument",
            target: BuiltinDocumentationLinkTarget::Builtin("tokenizedDocument")
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
