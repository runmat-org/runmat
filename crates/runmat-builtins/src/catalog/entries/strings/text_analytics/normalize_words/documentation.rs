use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Supported forms", paragraphs: &[
        "updatedDocuments = normalizeWords(documents)\nupdatedWords = normalizeWords(words)\nupdatedWords = normalizeWords(words, 'Language', language)\n___ = normalizeWords(___, 'Style', style)",
        "`updatedDocuments` is a tokenizedDocument compatibility object. `updatedWords` preserves string-array shape and cell-array shape. Character arrays preserve row count and are repadded after normalization.",
        "`documents` must be a RunMat `tokenizedDocument` object.",
        "`words` may be a string scalar, string array, character vector, character array, or cell array containing scalar string or character-vector elements.",
        "`language` must be `\"en\"` or `\"de\"` for standalone word input.",
        "`style` must be `\"stem\"` or `\"lemma\"`; `\"lemma\"` supports English in this compatibility slice.",
    ] },
    BuiltinDocumentationSection {
        heading: "Text forms and controls",
        paragraphs: &[
            "normalizeWords(words) reduces each standalone word to a root form using English stemming by default. The Language option accepts en or de for standalone input; Style accepts stem or lemma, with lightweight English lemmatization for words and tokenized documents.",
            "String arrays retain their shape. Character vectors and arrays remain character arrays; each row is one word and rows are padded to a common width after normalization. Cell arrays preserve shape and preserve each scalar string versus character-vector element type. Each element must contain one word; multi-word elements raise RunMat:normalizeWords:InvalidInput. Missing strings are preserved.",
            "normalizeWords(documents) retains tokenizedDocument shape and metadata. It normalizes tokens of derived type letters or other, while preserving punctuation, URLs, hashtags, mentions, and emoticons. Document language comes from its Language metadata, so a Language name-value option applies only to standalone words.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Compatibility and execution",
        paragraphs: &[
            "English stemming is Porter-style. German stemming is conservative for common preprocessing, not exact release-to-release dictionary identity. Japanese and Korean MeCab-backed token details, full dictionary and part-of-speech lemmatization, and exact MathWorks dictionary identity remain open.",
            "Numeric, logical, symbolic, and provider-resident values are not word inputs and reject before gather or provider access. The builtin uses host text processing and has no GPU kernel or fusion path.",
        ],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "stem-string-array", title: "Stem a string array",
        program: "words = [\"a\" \"strongly\" \"worded\" \"collection\" \"of\" \"words\"]; newWords = normalizeWords(words)",
        display_output: Some("newWords = [\"a\" \"strongli\" \"word\" \"collect\" \"of\" \"word\"]"),
        compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(newWords), [1 6])); assert(isequal(newWords, [\"a\" \"strongli\" \"word\" \"collect\" \"of\" \"word\"]));" },
    },
    BuiltinExample {
        id: "lemma-english", title: "Lemmatize English words",
        program: "words = [\"building\" \"has\" \"floors\"]; newWords = normalizeWords(words, \"Style\", \"lemma\")",
        display_output: Some("newWords = [\"build\" \"have\" \"floor\"]"),
        compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(newWords, [\"build\" \"have\" \"floor\"]));" },
    },
    BuiltinExample {
        id: "stem-german", title: "Stem German words",
        program: "words = [\"Morgen\"; \"guter\"]; newWords = normalizeWords(words, \"Language\", \"de\")",
        display_output: Some("newWords = [\"morg\"; \"gut\"]"),
        compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(newWords), [2 1])); assert(isequal(newWords, [\"morg\"; \"gut\"]));" },
    },
    BuiltinExample {
        id: "documents", title: "Normalize tokenized documents",
        program: "documents = tokenizedDocument(\"a strongly worded collection of words\"); newDocuments = normalizeWords(documents)",
        display_output: Some("`newDocuments` is a tokenizedDocument containing `a strongli word collect of word`."),
        compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(newDocuments), size(documents)));" },
    },
];
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/text_analytics/normalize.rs"),
};
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("normalizeWords"), slug: Some("normalizeWords"),
    summary: "Stem or lemmatize words and tokenized documents.",
    description: "normalizeWords reduces standalone words to root forms or updates RunMat tokenizedDocument compatibility objects while retaining their container shape.",
    keywords: &["normalizeWords", "stem", "lemma", "lemmatize", "text analytics", "words"],
    related: &["stopWords", "erasePunctuation", "extractHTMLText"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does normalizeWords support tokenizedDocument input?", answer: "Yes, for RunMat's lightweight English and German objects. It normalizes word-like tokens and preserves punctuation and complex tokens." },
        BuiltinDocumentationFaq { question: "Can standalone words use Japanese or Korean?", answer: "No. Those forms require the remaining MeCab-backed tokenization work." },
        BuiltinDocumentationFaq { question: "Does normalizeWords execute on the GPU?", answer: "No. It operates on host text and has no provider kernel." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "stopWords", target: BuiltinDocumentationLinkTarget::Builtin("stopWords") },
        BuiltinDocumentationLink { label: "erasePunctuation", target: BuiltinDocumentationLinkTarget::Builtin("erasePunctuation") },
        BuiltinDocumentationLink { label: "extractHTMLText", target: BuiltinDocumentationLinkTarget::Builtin("extractHTMLText") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "forms and normalization rules", location: "builtins::strings::text_analytics::normalize::tests" }],
        notes: &["German dictionary identity, MeCab-backed Japanese/Korean details, and exact MATLAB normalization parity remain open."],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
