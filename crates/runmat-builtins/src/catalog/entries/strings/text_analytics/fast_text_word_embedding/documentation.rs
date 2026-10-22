use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("fastTextWordEmbedding"),
    slug: Some("fastTextWordEmbedding"),
    summary: "Return a compact fastText-style word embedding compatibility model.",
    description: "`fastTextWordEmbedding` returns a RunMat `wordEmbedding` compatibility object with `Dimension` 300 and a compact built-in English vocabulary.",
    keywords: &[
        "fastTextWordEmbedding",
        "wordEmbedding",
        "text analytics",
        "fastText",
        "pretrained",
    ],
    related: &[
        "word2vec",
        "vec2word",
        "doc2sequence",
        "readWordEmbedding",
        "trainWordEmbedding",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported form",
            paragraphs: &[
                "emb = fastTextWordEmbedding",
                "`emb` is a `wordEmbedding` compatibility object.",
                "No input arguments are accepted.",
                "The compact model is deterministic and bundled with RunMat.",
                "Full pretrained-corpus parity is not yet implemented.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "The supported form is the zero-argument `emb = fastTextWordEmbedding` form.",
                "The returned object exposes `Dimension` and `Vocabulary` and stores dense vectors in the same implementation-detail property used by `word2vec`, `vec2word`, and `doc2sequence`.",
                "The compact model includes curated vectors for common examples such as `Italy - Rome + Paris`, which maps back to `France` with `vec2word`.",
                "RunMat does not download MathWorks support packages. The full 1-million-word fastText English 16 Billion Token model remains tracked in the broader Text Analytics compatibility issue.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &[
                "`fastTextWordEmbedding` constructs host wordEmbedding metadata and dense vectors; it does not use provider kernels.",
                "`fastTextWordEmbedding` performs host object construction and has no provider kernel.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Creating a word embedding model",
            program: "emb = fastTextWordEmbedding",
            display_output: Some("`emb.Dimension` is 300."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(strcmp(class(emb), \"wordEmbedding\")); assert(emb.Dimension == 300); assert(~isempty(emb.Vocabulary));",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Use With Embedding Lookup",
            program: "emb = fastTextWordEmbedding;\nitaly = word2vec(emb,\"Italy\");\nrome = word2vec(emb,\"Rome\");\nparis = word2vec(emb,\"Paris\");\nword = vec2word(emb, italy - rome + paris)",
            display_output: Some("`word` is `\"France\"` for the bundled compact model."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(strcmp(word, \"France\")); assert(isequal(size(italy), [1 300]));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does this include MATLAB's full pretrained fastText support-package corpus?",
            answer: "No. RunMat currently provides a compact compatibility model so scripts can construct and use a `wordEmbedding` object. Full support-package corpus parity remains open.",
        },
        BuiltinDocumentationFaq {
            question: "Can the result be used with `word2vec`, `vec2word`, and `doc2sequence`?",
            answer: "Yes. The returned object uses the same RunMat `wordEmbedding` representation as `readWordEmbedding` and `trainWordEmbedding`.",
        },
        BuiltinDocumentationFaq {
            question: "Does `fastTextWordEmbedding` execute on the GPU?",
            answer: "No. It constructs host text/model metadata and dense vectors.",
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
            label: "doc2sequence",
            target: BuiltinDocumentationLinkTarget::Builtin("doc2sequence")
        },
        BuiltinDocumentationLink {
            label: "readWordEmbedding",
            target: BuiltinDocumentationLinkTarget::Builtin("readWordEmbedding")
        },
        BuiltinDocumentationLink {
            label: "trainWordEmbedding",
            target: BuiltinDocumentationLinkTarget::Builtin("trainWordEmbedding")
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
