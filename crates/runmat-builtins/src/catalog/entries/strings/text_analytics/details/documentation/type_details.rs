use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Supported forms", paragraphs: &[
        "updatedDocuments = addTypeDetails(documents)\nupdatedDocuments = addTypeDetails(documents, 'DiscardKnownValues', tf)\nupdatedDocuments = addTypeDetails(documents, 'TopLevelDomains', domains)",
        "`updatedDocuments` is a `tokenizedDocument` compatibility object.",
        "`documents` must be a RunMat `tokenizedDocument` object.",
        "`tf` must be a logical scalar; numeric and integer 0/1 values are rejected.",
        "`domains` must contain one or more valid top-level-domain labels.",
        "Use `tokenDetails(updatedDocuments)` to view the resulting type table.",
    ] },
    BuiltinDocumentationSection { heading: "Token types", paragraphs: &[
        "addTypeDetails(documents) detects token types in a RunMat tokenizedDocument object. If details are absent, one type is derived for each token. DiscardKnownValues defaults to logical false, preserving known types and filling missing or unknown values; logical true recomputes every type.",
        "TopLevelDomains accepts a character vector, string array, or cell array of character vectors. It overrides the document's stored TLD list for web-address recognition and is stored on the returned object.",
        "The lightweight classifier recognizes letters, digits, punctuation, email-address, web-address, hashtag, at-mention, emoticon, emoji, and other. Custom token-type tables, full MathWorks topLevelDomains corpus parity, and exact URL-classifier parity remain open.",
    ] },
    BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
        "The builtin classifies host token strings and updates object metadata; it has no provider kernel or fusion path.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "pretokenized", title: "Add types to pre-tokenized documents",
        program: "documents = tokenizedDocument([\"For\" \"more\" \"information\" \",\"], \"TokenizeMethod\", \"none\"); updated = addTypeDetails(documents); tdetails = tokenDetails(updated)",
        display_output: Some("tdetails.Type includes letters and punctuation"),
        compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(~isempty(tdetails.Type));" },
    },
    BuiltinExample { id: "recompute", title: "Recompute known types",
        program: "documents = tokenizedDocument(\"For more information\"); updated = addTypeDetails(documents, \"DiscardKnownValues\", true)",
        display_output: Some("updated has refreshed token type details"),
        compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(updated), size(documents)));" },
    },
    BuiltinExample { id: "custom-domains", title: "Configure web-address domains",
        program: "documents = tokenizedDocument(\"example.zz\", \"TokenizeMethod\", \"none\"); updated = addTypeDetails(documents, \"TopLevelDomains\", \"zz\"); tdetails = tokenDetails(updated)",
        display_output: Some("tdetails.Type contains web-address"),
        compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(any(strcmp(tdetails.Type, \"web-address\")));" },
    },
];
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/text_analytics/details.rs"),
};
pub(in super::super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("addTypeDetails"), slug: Some("addTypeDetails"),
    summary: "Detect token types in tokenized documents.",
    description: "addTypeDetails adds token-type metadata to a RunMat tokenizedDocument object and can override top-level domains used for web-address recognition.",
    keywords: &["addTypeDetails", "tokenizedDocument", "text analytics", "token type", "TopLevelDomains", "token types"],
    related: &["tokenDetails", "tokenizedDocument", "addSentenceDetails"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does addTypeDetails support TopLevelDomains?", answer: "Yes. Character, string-array, and cell-string lists customize web-address detection; full default corpus parity remains open." },
        BuiltinDocumentationFaq { question: "Does addTypeDetails execute on the GPU?", answer: "No. It classifies host token strings and updates metadata." },
    ],
    links: &[SOURCE], media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "type classification and options", location: "builtins::strings::text_analytics::details::tests" }],
        notes: &["Custom type tables and exact MathWorks TLD/URL classifier parity remain open."],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
