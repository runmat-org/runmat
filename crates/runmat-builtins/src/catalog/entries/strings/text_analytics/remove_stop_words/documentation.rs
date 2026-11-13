use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,

    title: Some("removeStopWords"),

    slug: Some("removeStopWords"),

    summary: "Remove stop words from tokenized documents.",

    description: "`removeStopWords(documents)` removes stop words from RunMat `tokenizedDocument` compatibility objects using the object's language metadata and the same compatibility lists returned by `stopWords`.",

    keywords: &["removeStopWords", "stop words", "tokenizedDocument", "text analytics", "filter"],
    related: &["tokenizedDocument", "stopWords", "normalizeWords", "removeShortWords"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported forms",
            paragraphs: &["newDocuments = removeStopWords(documents)\nnewDocuments = removeStopWords(documents,'IgnoreCase',false)", "`newDocuments` is a tokenizedDocument compatibility object.", "`documents` must be a RunMat `tokenizedDocument` object.", "`IgnoreCase` must be a logical scalar; the default is `true`.", "The document object's `Language` metadata selects the stop-word list."],
        },
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`documents` must be a RunMat `tokenizedDocument` object created by `tokenizedDocument`.",
                "Numeric and resident tensor inputs, including every integer class, are not text documents and reject before provider access rather than being gathered or converted to text.",
                "`removeStopWords(documents)` is case-insensitive by default.",
                "`removeStopWords(documents,'IgnoreCase',false)` removes only tokens that match the stop-word list case exactly.",
                "The result preserves `Shape`, `TokenizeMethod`, and `Language` metadata while recomputing document lengths and vocabulary.",
                "RunMat removes stop words from derived `letters` and `other` token types. Punctuation, URLs, hashtags, mentions, and emoticons are preserved by this helper.",
                "English and German tokenized documents are supported by RunMat's current tokenizer. Japanese and Korean stop-word lists exist for `stopWords`, but MeCab-backed Japanese/Korean tokenized-document support remains tracked by the broader Text Analytics compatibility issue.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &["`removeStopWords` filters host text metadata and does not use provider kernels.", "`removeStopWords` filters host text metadata and has no provider kernel."],
        },
        BuiltinDocumentationSection {
            heading: "Implementation boundary",
            paragraphs: &["The host path builds a case-sensitive or case-folded stop-word set for the document language and filters supported token types. It does not add a general language-model engine or provider execution."],
        },
        BuiltinDocumentationSection {
            heading: "Finite host allocation",
            paragraphs: &["The current implementation requires an Unbounded host allocation policy. Under Managed finite allocation, a shared family guard rejects before gather, model cloning, mutation or filesystem work because complete text-model scratch and metadata ownership is not yet implemented. Authority is checked again after suspension and before publication, and failures are terminal rather than permission to replay. This refusal is an implementation frontier, not completion of finite resource support."],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Remove English Stop Words",
            program: "documents = tokenizedDocument([\"an example of a short sentence\"; \"a second short sentence\"]);\nnewDocuments = removeStopWords(documents)",
            display_output: Some("`newDocuments` contains `example short sentence` and `second short sentence`."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(newDocuments.Vocabulary, [\"example\" \"short\" \"sentence\" \"second\"])); assert(isequal(newDocuments.DocumentLengths, [3; 3]));",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Remove Stop Words Case Sensitively",
            program: "documents = tokenizedDocument([\"The\" \"the\" \"word\"], \"TokenizeMethod\", \"none\");\nnewDocuments = removeStopWords(documents, \"IgnoreCase\", false)",
            display_output: Some("`newDocuments` contains `The word`."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(newDocuments.Vocabulary, [\"The\" \"word\"])); assert(newDocuments.DocumentLengths == 2);",
            },
        },
    ],
    example_exemption: None,

    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does `removeStopWords` support raw strings?",
            answer: "No. Convert raw text with `tokenizedDocument` first.",
        },
        BuiltinDocumentationFaq {
            question: "Can I provide a custom word list?",
            answer: "No. Use of custom lists belongs to `removeWords`, which remains tracked by the broader Text Analytics compatibility issue.",
        },
        BuiltinDocumentationFaq {
            question: "Does `removeStopWords` execute on the GPU?",
            answer: "No. It filters host text metadata and has no `runmat-accelerate` provider path.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "tokenizedDocument",
            target: BuiltinDocumentationLinkTarget::Builtin("tokenizedDocument"),
        },
        BuiltinDocumentationLink {
            label: "stopWords",
            target: BuiltinDocumentationLinkTarget::Builtin("stopWords"),
        },
        BuiltinDocumentationLink {
            label: "normalizeWords",
            target: BuiltinDocumentationLinkTarget::Builtin("normalizeWords"),
        },
        BuiltinDocumentationLink {
            label: "removeShortWords",
            target: BuiltinDocumentationLinkTarget::Builtin("removeShortWords"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/strings/text_analytics/documents.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/text_analytics/documents.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "unit",
            location: "builtins::strings::text_analytics::documents::tests",
        }],
        notes: &[],
    },
    introduced: None,

    status: Some(BuiltinDocumentationStatus::Partial),
};
