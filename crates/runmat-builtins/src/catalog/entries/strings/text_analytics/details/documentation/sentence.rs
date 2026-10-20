use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Supported forms", paragraphs: &[
        "updatedDocuments = addSentenceDetails(documents)\nupdatedDocuments = addSentenceDetails(documents, 'Abbreviations', abbreviations)\nupdatedDocuments = addSentenceDetails(documents, 'Starters', starters)\nupdatedDocuments = addSentenceDetails(documents, 'DiscardKnownValues', tf)",
        "`updatedDocuments` is a `tokenizedDocument` compatibility object.",
        "`documents` must be a RunMat `tokenizedDocument` object.",
        "`abbreviations` can be text or a table with `Abbreviation` and `Usage` variables.",
        "`starters` must be a text or categorical vector.",
        "`tf` must be a logical scalar; numeric and integer 0/1 values are rejected.",
        "Use `tokenDetails(updatedDocuments)` to view the resulting `SentenceNumber` table column.",
    ] },
    BuiltinDocumentationSection { heading: "Sentence numbers", paragraphs: &[
        "addSentenceDetails(documents) detects sentence boundaries in a RunMat tokenizedDocument object and stores one SentenceNumber per token. tokenDetails shows the resulting column.",
        "Terminating punctuation includes period, exclamation, question mark, and common Japanese/Korean equivalents. Single-letter initialisms such as U.S.A. are treated as regular abbreviations and do not split internally.",
        "Abbreviations accepts a character vector, string array, cell array of character vectors, or a table with Abbreviation and Usage variables. Text lists are regular abbreviations; table Usage may be text or categorical regular, inner, reference, or unit.",
        "Starters accepts a character vector, string array, cell array of character vectors, or categorical vector. It controls whether regular abbreviations break before capitalized starter words. The default list combines English stopWords with common sentence starters.",
        "DiscardKnownValues defaults to logical false and preserves complete positive integer sentence-number details. Logical true recomputes every number. The detector is deterministic and token-level; exact MathWorks statistical boundaries, emoji-after-terminator placement, line provenance, and MeCab behavior remain open.",
    ] },
    BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
        "The builtin updates host tokenizedDocument metadata and has no provider kernel or fusion path.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "sentences", title: "Add sentence details",
        program: "documents = tokenizedDocument([\"This is an example document. It has two sentences.\"; \"This document has one sentence.\"]); updated = addSentenceDetails(documents); tdetails = tokenDetails(updated)",
        display_output: Some("tdetails includes SentenceNumber"),
        compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(~isempty(tdetails.SentenceNumber));" },
    },
    BuiltinExample { id: "custom-abbreviations", title: "Use custom abbreviations",
        program: "documents = tokenizedDocument(\"Book an appt. We'll meet then.\"); updated = addSentenceDetails(documents, \"Abbreviations\", \"appt\", \"Starters\", \"we'll\")",
        display_output: Some("appt. is treated as a regular abbreviation with the configured starter"),
        compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(updated), size(documents)));" },
    },
    BuiltinExample { id: "recompute", title: "Recompute sentence numbers",
        program: "documents = tokenizedDocument(\"One. Two.\"); updated = addSentenceDetails(documents, \"DiscardKnownValues\", true)",
        display_output: Some("updated has refreshed sentence-number details"),
        compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(updated), size(documents)));" },
    },
];
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/text_analytics/details.rs"),
};
pub(in super::super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("addSentenceDetails"), slug: Some("addSentenceDetails"),
    summary: "Detect sentence boundaries in tokenized documents.",
    description: "addSentenceDetails stores sentence-number metadata on a RunMat tokenizedDocument object.",
    keywords: &["addSentenceDetails", "tokenizedDocument", "text analytics", "sentence", "abbreviations", "sentence numbers", "token details"],
    related: &["tokenDetails", "tokenizedDocument", "stopWords"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does addSentenceDetails support abbreviation tables?", answer: "Yes. The table needs Abbreviation and Usage variables, with text or categorical regular, inner, reference, or unit usage." },
        BuiltinDocumentationFaq { question: "Does addSentenceDetails execute on the GPU?", answer: "No. It updates host tokenizedDocument metadata." },
    ],
    links: &[SOURCE, BuiltinDocumentationLink { label: "addTypeDetails", target: BuiltinDocumentationLinkTarget::Builtin("addTypeDetails") }], media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "sentence details and controls", location: "builtins::strings::text_analytics::details::tests" }],
        notes: &["Statistical boundary, line provenance, emoji placement, and MeCab parity remain open."],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
