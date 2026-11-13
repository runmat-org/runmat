use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("wordEncoding"),
    slug: Some("wordEncoding"),
    summary: "Create a word encoding object that maps words to indices and back.",
    description: "`wordEncoding(documents)` creates a `wordEncoding` object from tokenized documents. `wordEncoding(words)` creates one from a word vector.",
    keywords: &[
        "wordEncoding",
        "text analytics",
        "vocabulary",
        "indices",
    ],
    related: &[
        "word2ind",
        "ind2word",
        "isVocabularyWord",
        "doc2sequence",
        "tokenizedDocument",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported forms",
            paragraphs: &[
                "enc = wordEncoding(documents)\nenc = wordEncoding(words)\nenc = wordEncoding(documents, Name, Value)",
                "`enc` is a `wordEncoding` object.",
                "`Order` must be `\"first-seen\"` or `\"frequency\"` and is only accepted for tokenizedDocument input.",
                "`MaxNumWords` must be a positive integer-valued double or `Inf` and is only accepted for tokenizedDocument input. Native integer storage for this option requires `runmat` mode.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`documents` must be a `tokenizedDocument` object.",
                "`words` may be a string scalar, string array, character vector, character matrix, or cell array of scalar text. A character vector is treated as one word.",
                "The returned object exposes `NumWords` and `Vocabulary`. `Vocabulary` is a 1-by-NumWords string array.",
                "`Order` accepts `\"first-seen\"` and `\"frequency\"` for tokenizedDocument input. The default is first-seen order.",
                "For frequency ordering, RunMat sorts by descending total count and uses first-seen order as a deterministic tie-breaker.",
                "`MaxNumWords` accepts a positive integer-valued double or `Inf` for tokenizedDocument input. The option is applied after ordering.",
                "A native integer scalar for `MaxNumWords` is a RunMat language extension. It is accepted in `runmat` mode and decoded exactly as a positive structural count.",
                "RunMat rejects `Order` and `MaxNumWords` when the first input is a word vector, matching the documented MATLAB syntax.",
                "First-class MATLAB object-array identity semantics are not required for this scalar model object and remain covered by the broader object infrastructure work.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &[
                "`wordEncoding` builds host text metadata and has no provider-resident numeric kernel.",
                "`wordEncoding` constructs host text and object metadata and does not launch provider kernels.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Create Encoding From Documents",
            program: "documents = tokenizedDocument([\"alpha beta alpha\"; \"gamma beta\"]);\nenc = wordEncoding(documents)",
            display_output: Some("`enc.Vocabulary` is `[\"alpha\" \"beta\" \"gamma\"]` for first-seen order."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(enc.NumWords == 3); assert(isequal(enc.Vocabulary, [\"alpha\" \"beta\" \"gamma\"]));",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Sort By Frequency",
            program: "documents = tokenizedDocument([\"alpha beta alpha\"; \"gamma beta\"]);\nenc = wordEncoding(documents, \"Order\", \"frequency\", \"MaxNumWords\", 2)",
            display_output: Some("`enc.Vocabulary` contains the two most frequent words."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(enc.NumWords == 2); assert(isequal(enc.Vocabulary, [\"alpha\" \"beta\"]));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Can the result be used with `doc2sequence`?",
            answer: "Yes. `doc2sequence(enc, documents)` returns cell arrays of 1-by-S word-index vectors.",
        },
        BuiltinDocumentationFaq {
            question: "Does `wordEncoding` execute on the GPU?",
            answer: "No. It constructs host text metadata and does not use provider kernels.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "MATLAB language extensions",
            target: BuiltinDocumentationLinkTarget::Documentation("/docs/runtime/getting-started/language-extensions")
        },
        BuiltinDocumentationLink {
            label: "word2ind",
            target: BuiltinDocumentationLinkTarget::Builtin("word2ind")
        },
        BuiltinDocumentationLink {
            label: "ind2word",
            target: BuiltinDocumentationLinkTarget::Builtin("ind2word")
        },
        BuiltinDocumentationLink {
            label: "isVocabularyWord",
            target: BuiltinDocumentationLinkTarget::Builtin("isVocabularyWord")
        },
        BuiltinDocumentationLink {
            label: "doc2sequence",
            target: BuiltinDocumentationLinkTarget::Builtin("doc2sequence")
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
                label: "`crates/runmat-runtime/src/builtins/strings/text_analytics/encoding.rs`",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/text_analytics/encoding.rs")
            },
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::strings::text_analytics::encoding::tests",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
