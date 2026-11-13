use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,

    title: Some("removeWords"),

    slug: Some("removeWords"),

    summary: "Remove selected words from tokenized documents or bag-of-words models.",

    description: "`removeWords(documents, words)` removes selected tokens from RunMat `tokenizedDocument` compatibility objects. `removeWords(bag, words)` removes matching vocabulary columns from RunMat `bagOfWords` objects. `words` can also be numeric 1-based vocabulary indices or a logical mask the same length as the object's vocabulary.",

    keywords: &["removeWords", "tokenizedDocument", "bagOfWords", "text analytics", "filter"],
    related: &["tokenizedDocument", "bagOfWords", "stopWords", "removeShortWords", "removeLongWords"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported forms",
            paragraphs: &["newDocuments = removeWords(documents, words)\nnewBag = removeWords(bag, words)\nnewDocuments = removeWords(___, 'IgnoreCase', true)\nnewDocuments = removeWords(documents, idx)\nnewBag = removeWords(bag, idx)", "`newDocuments` is a `tokenizedDocument` compatibility object. `newBag` is a `bagOfWords` compatibility object.", "Input objects must be RunMat Text Analytics compatibility objects from this module.", "Numeric indices are 1-based positive integers into the object's vocabulary.", "Logical masks must have the same number of elements as the object's vocabulary."],
        },
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &["`documents` must be a RunMat `tokenizedDocument` object created by `tokenizedDocument`.", "`bag` must be a RunMat `bagOfWords` object created by `bagOfWords`.", "`words` can be a string vector, character vector, or cell array of character vectors. A character vector is treated as one word.", "`idx` can be a numeric vector of positive integer vocabulary positions from any integer class, or a logical array whose length matches the object's `Vocabulary`; native positions remain exact through bounds and duplicate checks.", "`removeWords(___, 'IgnoreCase', true)` removes string-selected or index-selected vocabulary words without case sensitivity. The default is case-sensitive.", "For tokenized documents, the result preserves `Shape`, `TokenizeMethod`, and `Language` metadata while recomputing document lengths and vocabulary.", "For bag-of-words models, the result preserves document row count and removes matching count columns."],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &["`removeWords` filters host text-model objects. Automatic residency gathers transparently; explicit gpuArray host fallback is a declared RunMat extension.", "`removeWords` filters host text/model objects and has no provider kernel."],
        },
        BuiltinDocumentationSection {
            heading: "Implementation boundary",
            paragraphs: &["Selectors are validated against host vocabulary order before filtering tokens or dense bag count columns. Explicit resident fallback remains extension-gated; no provider text-filter kernel is implemented."],
        },
        BuiltinDocumentationSection {
            heading: "Finite host allocation",
            paragraphs: &["The current implementation requires an Unbounded host allocation policy. Under Managed finite allocation, a shared family guard rejects before gather, model cloning, mutation or filesystem work because complete text-model scratch and metadata ownership is not yet implemented. Authority is checked again after suspension and before publication, and failures are terminal rather than permission to replay. This refusal is an implementation frontier, not completion of finite resource support."],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Remove Words From Documents",
            program: "documents = tokenizedDocument([\"an example of a short sentence\"; \"a second short sentence\"]);\nnewDocuments = removeWords(documents, [\"short\" \"second\"])",
            display_output: Some("`newDocuments` contains `an example of a sentence` and `a sentence`."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(newDocuments.Vocabulary, [\"an\" \"example\" \"of\" \"a\" \"sentence\"])); assert(isequal(newDocuments.DocumentLengths, [5; 2]));",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Remove Words By Vocabulary Index",
            program: "documents = tokenizedDocument([\"I love MATLAB\"; \"I love MathWorks\"]);\nnewDocuments = removeWords(documents, [1 3])",
            display_output: Some("`newDocuments` removes the first and third vocabulary entries."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(newDocuments.Vocabulary, [\"love\" \"MathWorks\"])); assert(isequal(newDocuments.DocumentLengths, [1; 2]));",
            },
        },
        BuiltinExample {
            id: "retained-3",
            title: "Remove Words From A Bag",
            program: "documents = tokenizedDocument([\"an example of a short sentence\"; \"a second short sentence\"]);\nbag = bagOfWords(documents);\nnewBag = removeWords(bag, stopWords)",
            display_output: Some("`newBag` removes count columns for the selected vocabulary words."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(newBag.Vocabulary, [\"example\" \"short\" \"sentence\" \"second\"])); assert(isequal(newBag.Counts, [1 1 1 0; 0 1 1 1]));",
            },
        },
    ],
    example_exemption: None,

    faqs: &[
        BuiltinDocumentationFaq {
            question: "Is `removeWords` case-sensitive?",
            answer: "Yes by default. Use `removeWords(___, 'IgnoreCase', true)` for case-insensitive matching.",
        },
        BuiltinDocumentationFaq {
            question: "Does a logical index mask need to match the vocabulary length?",
            answer: "Yes. RunMat rejects logical masks whose length differs from the object's vocabulary length.",
        },
        BuiltinDocumentationFaq {
            question: "Does `removeWords` execute on the GPU?",
            answer: "No. It filters host text/model metadata and has no `runmat-accelerate` provider path.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "tokenizedDocument",
            target: BuiltinDocumentationLinkTarget::Builtin("tokenizedDocument"),
        },
        BuiltinDocumentationLink {
            label: "bagOfWords",
            target: BuiltinDocumentationLinkTarget::Builtin("bagOfWords"),
        },
        BuiltinDocumentationLink {
            label: "stopWords",
            target: BuiltinDocumentationLinkTarget::Builtin("stopWords"),
        },
        BuiltinDocumentationLink {
            label: "removeShortWords",
            target: BuiltinDocumentationLinkTarget::Builtin("removeShortWords"),
        },
        BuiltinDocumentationLink {
            label: "removeLongWords",
            target: BuiltinDocumentationLinkTarget::Builtin("removeLongWords"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/strings/text_analytics/documents.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/text_analytics/documents.rs"),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::strings::text_analytics::documents::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "integration",
                location: "crates/runmat-vm/tests/integer_label_and_text_control_semantics.rs",
            },
        ],
        notes: &[],
    },
    introduced: None,

    status: Some(BuiltinDocumentationStatus::Partial),
};
