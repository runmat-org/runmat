use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Supported forms", paragraphs: &[
        "words = stopWords\nwords = stopWords('Language', language)",
        "`words` is an `N x 1` string array.",
        "`language` must be `\"en\"`, `\"ja\"`, `\"de\"`, or `\"ko\"`.",
        "The option name is case-insensitive and must be `Language`.",
    ] },
    BuiltinDocumentationSection {
        heading: "Languages and output",
        paragraphs: &[
            "stopWords returns an N-by-1 host string array. With no inputs it returns the English stop-word list. The case-insensitive Language name-value option selects en, ja, de, or ko.",
            "The lists cover common text-preprocessing words. Exact vocabulary identity across MathWorks releases is not guaranteed; code requiring a fixed vocabulary should retain its own list.",
            "removeStopWords consumes these lists when filtering RunMat tokenizedDocument objects. Custom token deletion remains part of the broader Text Analytics object surface.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution",
        paragraphs: &[
            "The builtin accepts scalar string or character-vector controls and rejects numeric and resident inputs before provider access. It returns host-resident string metadata; there is no GPU kernel, reduction, broadcast, or fusion path.",
            "Under a finite host allocation policy, the result's collection, shape, and each nonempty UTF-8 payload are admitted before construction and remain owned until the last alias retires.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "english-default",
        title: "Get English stop words",
        program: "words = stopWords",
        display_output: Some("words = N×1 string\n    \"a\"\n    \"about\"\n    \"above\"\n    ..."),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isstring(words)); assert(size(words,2) == 1); assert(any(strcmp(words, \"a\")));",
        },
    },
    BuiltinExample {
        id: "german-language",
        title: "Get German stop words",
        program: "words = stopWords('Language','de')",
        display_output: Some("words = N×1 string\n    \"ab\"\n    \"aber\"\n    \"alle\"\n    ..."),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isstring(words)); assert(size(words,2) == 1); assert(any(strcmp(words, \"aber\")));",
        },
    },
    BuiltinExample {
        id: "custom-filter",
        title: "Extend a stop-word list",
        program: "words = stopWords; custom = [words; \"runmat\"; \"matlab\"]",
        display_output: Some("custom = column string array including the custom terms"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(size(custom,2) == 1); assert(strcmp(custom(end-1), \"runmat\")); assert(strcmp(custom(end), \"matlab\"));",
        },
    },
];

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/text_analytics/stopwords.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("stopWords"),
    slug: Some("stopWords"),
    summary: "Return common stop words for supported Text Analytics languages.",
    description: "stopWords returns a column string array of common stop words in English, Japanese, German, or Korean.",
    keywords: &["stopWords", "stop words", "text analytics", "language", "string"],
    related: &["removeStopWords", "erasePunctuation", "string", "strings"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Which language codes are accepted?", answer: "en, ja, de, and ko are accepted. Other codes raise RunMat:stopWords:InvalidInput." },
        BuiltinDocumentationFaq { question: "Is this tied to tokenizedDocument?", answer: "stopWords is a standalone list function; removeStopWords uses the same lists for tokenized-document filtering." },
        BuiltinDocumentationFaq { question: "Does stopWords execute on the GPU?", answer: "No. It returns host string metadata and has no runmat-accelerate provider path." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "erasePunctuation", target: BuiltinDocumentationLinkTarget::Builtin("erasePunctuation") },
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "strings", target: BuiltinDocumentationLinkTarget::Builtin("strings") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "language forms and errors", location: "builtins::strings::text_analytics::stopwords::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "static inference", location: "crates/runmat-builtins/src/catalog/entries/strings/text_analytics/stopwords/inference.rs" },
        ],
        notes: &["Exact MathWorks vocabulary identity across releases is not guaranteed."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
