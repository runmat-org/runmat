use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Supported forms", paragraphs: &[
        "updatedDocuments = addLemmaDetails(documents)\nupdatedDocuments = addLemmaDetails(documents, 'DiscardKnownValues', tf)",
        "`updatedDocuments` is a `tokenizedDocument` compatibility object.",
        "`documents` must be a RunMat `tokenizedDocument` object.",
        "`tf` must be a logical scalar; numeric and integer 0/1 values are rejected.",
        "Use `tokenDetails(updatedDocuments)` to view the resulting `Lemma` table column.",
    ] },
    BuiltinDocumentationSection {
        heading: "Lemma metadata",
        paragraphs: &[
            "addLemmaDetails(documents) adds lemma metadata to a RunMat tokenizedDocument object. tokenDetails shows the resulting Lemma table column.",
            "DiscardKnownValues defaults to false: known stored lemmas are preserved while missing or empty entries are filled. Set it to a logical true to recompute every lemma; numeric 0 or 1 is not a logical option.",
            "English uses RunMat's deterministic lightweight lemmatizer, shared with normalizeWords Style lemma. It handles common irregulars such as ran to run, plural reduction, and simple verb suffixes; it is not a full NLP model or a claim of exact model parity.",
            "Punctuation, numbers, URLs, email addresses, hashtags, mentions, emoji, and other non-word tokens retain their token text. Existing Japanese or Korean document metadata retains token text as lemma; public construction for those languages still requires MeCab-compatible tokenization. German documents reject because the documented form does not support German.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution",
        paragraphs: &[
            "The builtin updates host tokenizedDocument metadata and invokes no provider kernel or fusion path. Numeric and resident input values do not become text implicitly.",
        ],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "add-lemmas",
        title: "Add lemma details",
        program: "documents = tokenizedDocument([\"The dogs ran after the cat.\"; \"I am building a house.\"]); updated = addLemmaDetails(documents); tdetails = tokenDetails(updated)",
        display_output: Some("tdetails.Lemma includes dog, run, be, and build"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(any(strcmp(tdetails.Lemma, \"dog\"))); assert(any(strcmp(tdetails.Lemma, \"run\")));" },
    },
    BuiltinExample {
        id: "recompute-lemmas",
        title: "Recompute known lemmas",
        program: "documents = tokenizedDocument(\"The dogs ran.\"); updated = addLemmaDetails(documents, \"DiscardKnownValues\", true); tdetails = tokenDetails(updated)",
        display_output: Some("tdetails.Lemma includes dog and run"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(any(strcmp(tdetails.Lemma, \"run\")));" },
    },
];
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/text_analytics/lemmas.rs"),
};
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("addLemmaDetails"),
    slug: Some("addLemmaDetails"),
    summary: "Add lemma forms to tokenized documents.",
    description: "addLemmaDetails adds lemma metadata to a RunMat tokenizedDocument compatibility object.",
    keywords: &["addLemmaDetails", "tokenizedDocument", "text analytics", "lemma", "lemmatization"],
    related: &["tokenDetails", "tokenizedDocument", "normalizeWords"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does addLemmaDetails use a full NLP model?", answer: "No. RunMat uses a deterministic English lemmatizer; exact NLP-model parity remains open." },
        BuiltinDocumentationFaq { question: "Does addLemmaDetails execute on the GPU?", answer: "No. It updates host tokenizedDocument metadata." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "tokenDetails", target: BuiltinDocumentationLinkTarget::Builtin("tokenDetails") },
        BuiltinDocumentationLink { label: "tokenizedDocument", target: BuiltinDocumentationLinkTarget::Builtin("tokenizedDocument") },
        BuiltinDocumentationLink { label: "normalizeWords", target: BuiltinDocumentationLinkTarget::Builtin("normalizeWords") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "lemma details and options", location: "builtins::strings::text_analytics::lemmas::tests" }],
        notes: &["Japanese and Korean public tokenizedDocument construction remains dependent on MeCab-compatible tokenization; English model parity is not claimed."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
