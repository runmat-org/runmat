use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Supported forms", paragraphs: &[
        "updatedDocuments = addPartOfSpeechDetails(documents)\nupdatedDocuments = addPartOfSpeechDetails(documents, Name, Value)",
        "`updatedDocuments` is a `tokenizedDocument` compatibility object.",
        "`documents` must be a RunMat `tokenizedDocument` object.",
        "`Name, Value` pairs may include `RetokenizeMethod`, `Abbreviations`, and `DiscardKnownValues`.",
        "`DiscardKnownValues` must be a logical scalar; numeric and integer 0/1 values are rejected.",
        "Use `tokenDetails(updatedDocuments)` to view the resulting `PartOfSpeech` table column.",
    ] },
    BuiltinDocumentationSection { heading: "Part-of-speech metadata", paragraphs: &[
        "addPartOfSpeechDetails(documents) adds part-of-speech tags to a RunMat tokenizedDocument object. tokenDetails exposes the resulting PartOfSpeech column.",
        "RetokenizeMethod defaults to part-of-speech and handles common English contractions, wanna/gonna, ellipses, and dotted initialisms; none keeps existing tokens. Abbreviations are forwarded to the sentence-detail prepass when sentence details are missing.",
        "DiscardKnownValues defaults to logical false: existing known tags are preserved and missing entries are filled. Logical true recomputes every stored tag; numeric 0 or 1 is not accepted as a logical flag.",
        "RunMat uses deterministic English and German classifiers for pronouns, determiners, coordinate and subordinate conjunctions, adpositions, particles, auxiliary verbs, verbs, adverbs, adjectives, nouns, numerals, punctuation, and other tokens. Existing Japanese or Korean metadata uses token-class-aware fallback tags and a conservative noun fallback for letters. Public MeCab-backed construction and exact MathWorks statistical-model parity remain open.",
    ] },
    BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
        "The builtin retokenizes host text and updates host tokenizedDocument metadata. It has no provider kernel or fusion path.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "add-tags", title: "Add part-of-speech details",
        program: "documents = tokenizedDocument(\"The dogs are running.\"); updated = addPartOfSpeechDetails(documents); tdetails = tokenDetails(updated)",
        display_output: Some("tdetails.PartOfSpeech contains determiner, noun, auxiliary-verb, verb, and punctuation tags"),
        compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(~isempty(tdetails.PartOfSpeech));" },
    },
    BuiltinExample { id: "keep-tokens", title: "Keep existing tokens",
        program: "documents = tokenizedDocument(\"The dogs are running.\"); updated = addPartOfSpeechDetails(documents, \"RetokenizeMethod\", \"none\")",
        display_output: Some("updated has part-of-speech details without retokenization"),
        compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(updated), size(documents)));" },
    },
    BuiltinExample { id: "recompute-tags", title: "Recompute known tags",
        program: "documents = tokenizedDocument(\"The dogs run.\"); updated = addPartOfSpeechDetails(documents, \"DiscardKnownValues\", true)",
        display_output: Some("updated has refreshed part-of-speech details"),
        compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(updated), size(documents)));" },
    },
];
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/text_analytics/pos.rs"),
};
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("addPartOfSpeechDetails"), slug: Some("addPartOfSpeechDetails"),
    summary: "Add part-of-speech tags to tokenized documents.",
    description: "addPartOfSpeechDetails adds part-of-speech metadata to a RunMat tokenizedDocument compatibility object.",
    keywords: &["addPartOfSpeechDetails", "tokenizedDocument", "text analytics", "part of speech", "pos"],
    related: &["tokenDetails", "tokenizedDocument", "addSentenceDetails"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does addPartOfSpeechDetails use a full NLP model?", answer: "No. It uses deterministic compatibility classifiers; exact MathWorks model parity remains open." },
        BuiltinDocumentationFaq { question: "Does addPartOfSpeechDetails execute on the GPU?", answer: "No. It updates host tokenizedDocument metadata." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "tokenDetails", target: BuiltinDocumentationLinkTarget::Builtin("tokenDetails") },
        BuiltinDocumentationLink { label: "tokenizedDocument", target: BuiltinDocumentationLinkTarget::Builtin("tokenizedDocument") },
        BuiltinDocumentationLink { label: "addSentenceDetails", target: BuiltinDocumentationLinkTarget::Builtin("addSentenceDetails") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "POS controls, retokenization and tags", location: "builtins::strings::text_analytics::pos::tests" }],
        notes: &["MeCab-backed Japanese/Korean construction and exact statistical model parity remain open."],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
