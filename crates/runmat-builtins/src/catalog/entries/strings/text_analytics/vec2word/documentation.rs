use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,

    title: Some("vec2word"),

    slug: Some("vec2word"),

    summary: "Map embedding vectors to nearest vocabulary words.",

    description: "`vec2word(emb, M)` returns the nearest vocabulary word for each row of `M`. `[words, dist] = vec2word(emb, M, k)` returns the `k` nearest words and distances.",

    keywords: &["vec2word", "wordEmbedding", "text analytics", "nearest words"],
    related: &["readWordEmbedding", "word2vec"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported forms",
            paragraphs: &["words = vec2word(emb, M)\n[words, dist] = vec2word(emb, M, k)\n[words, dist] = vec2word(emb, M, k, 'Distance', distance)", "`words` is a string array. `dist` is a numeric matrix.", "`distance` must be `\"cosine\"` or `\"euclidean\"`.", "`M` columns must equal `emb.Dimension`.", "Native integer `M` and native integer `k` require RunMat language extensions to be enabled."],
        },
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &["`emb` must be a RunMat `wordEmbedding` compatibility object.", "`M` must be an `n`-by-`emb.Dimension` numeric matrix.", "`k` is an optional positive integer from `1` through the vocabulary size. The default is `1`.", "Native integer storage for `M` and native integer storage for `k` are independent RunMat language extensions. Integer query values must be exactly representable as double before distance calculation, while `k` is decoded as an exact structural value.", "The default distance metric is cosine distance. `vec2word(___, 'Distance', 'euclidean')` uses Euclidean distance.", "`words` is an `n`-by-`k` string array. `dist` is an `n`-by-`k` numeric matrix.", "For zero-norm query or vocabulary vectors under cosine distance, RunMat reports infinite distance so finite matches sort ahead of them."],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &["`vec2word` compares host-resident word embedding model data and has no provider kernel. Supported resident query data may gather automatically through its owning provider.", "`vec2word` compares host text-model storage and has no provider kernel. Supported resident query data may gather automatically through its owning provider."],
        },
        BuiltinDocumentationSection {
            heading: "Implementation boundary",
            paragraphs: &["The current host implementation scores the vocabulary and sorts candidate distances for each query row; it is not an approximate-nearest-neighbor index. Returned words and distances are host-owned, and successful resident input gathering does not imply a provider search kernel."],
        },
        BuiltinDocumentationSection {
            heading: "Finite host allocation",
            paragraphs: &["The current implementation requires an Unbounded host allocation policy. Under Managed finite allocation, a shared family guard rejects before gather, model cloning, mutation or filesystem work because complete text-model scratch and metadata ownership is not yet implemented. Authority is checked again after suspension and before publication, and failures are terminal rather than permission to replay. This refusal is an implementation frontier, not completion of finite resource support."],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Find Nearest Words",
            program: "emb = readWordEmbedding('vectors.vec');\nM = word2vec(emb, [\"king\" \"queen\"]);\nwords = vec2word(emb, M)",
            display_output: Some("`words` contains the nearest vocabulary word for each vector row."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
                id: BuiltinExampleFixtureId {
                    local_name: "vec2word-1",
                },
                root: BuiltinFilesystemRoot::IsolatedWorkspace,
                entries: &[BuiltinFilesystemEntry::File {
                    relative_path: "vectors.vec",
                    content: BuiltinFixtureContent::Utf8("3 2\nking 1 0\nqueen 0 1\nMATLAB 1 1\n"),
                }],
            }),
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(words, [\"king\"; \"queen\"]));",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Return Multiple Neighbors",
            program: "emb = readWordEmbedding(\"vectors.vec\"); M = word2vec(emb, [\"king\" \"queen\"]);\n[words, dist] = vec2word(emb, M, 3, \"Distance\", \"cosine\")",
            display_output: Some("`words` and `dist` have three columns."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::NativeFilesystem,
            fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
                id: BuiltinExampleFixtureId {
                    local_name: "vec2word-2",
                },
                root: BuiltinFilesystemRoot::IsolatedWorkspace,
                entries: &[BuiltinFilesystemEntry::File {
                    relative_path: "vectors.vec",
                    content: BuiltinFixtureContent::Utf8("3 2\nking 1 0\nqueen 0 1\nMATLAB 1 1\n"),
                }],
            }),
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(size(words), [2 3])); assert(isequal(size(dist), [2 3])); assert(isequal(words(:,1), [\"king\"; \"queen\"])); assert(all(abs(dist(:,1)) < 1e-12)); assert(all(all(diff(dist, 1, 2) >= -1e-12)));",
            },
        },
    ],
    example_exemption: None,

    faqs: &[
        BuiltinDocumentationFaq {
            question: "Which distances are supported?",
            answer: "RunMat supports `\"cosine\"` and `\"euclidean\"`.",
        },
        BuiltinDocumentationFaq {
            question: "Does `vec2word` execute on the GPU?",
            answer: "No. It compares the host-resident embedding model.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "readWordEmbedding",
            target: BuiltinDocumentationLinkTarget::Builtin("readWordEmbedding"),
        },
        BuiltinDocumentationLink {
            label: "word2vec",
            target: BuiltinDocumentationLinkTarget::Builtin("word2vec"),
        },
        BuiltinDocumentationLink {
            label: "MATLAB Language Extensions",
            target: BuiltinDocumentationLinkTarget::Documentation("/docs/runtime/getting-started/language-extensions"),
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
