use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Supported forms", paragraphs: &[
        "updatedDocuments = addEntityDetails(documents)\nupdatedDocuments = addEntityDetails(documents, Name, Value)",
        "`updatedDocuments` is a `tokenizedDocument` compatibility object.",
        "`documents` must be a RunMat `tokenizedDocument` object.",
        "`Name, Value` pairs may include `RetokenizeMethod`, `DiscardKnownValues`, and `Model`.",
        "`RetokenizeMethod` may be `\"entity\"` or `\"none\"`.",
        "`DiscardKnownValues` must be a logical scalar; numeric and integer 0/1 values are rejected.",
        "`Model` currently accepts `\"auto\"`; custom entity model objects remain tracked.",
        "Use `tokenDetails(updatedDocuments)` to view the resulting `Entity` table column.",
    ] },
    BuiltinDocumentationSection { heading: "Entity metadata and controls", paragraphs: &[
        "addEntityDetails(documents) adds named-entity tags to a RunMat tokenizedDocument object; tokenDetails exposes the Entity column. RetokenizeMethod, DiscardKnownValues, and Model are the supported name-value options.",
        "RetokenizeMethod defaults to entity and merges simple adjacent entity tokens such as multiword locations; none preserves existing tokens. DiscardKnownValues defaults to logical false, retaining known tags while filling missing entries. Logical true recomputes all tags and ignores malformed stored tags.",
        "Model auto selects the built-in deterministic compatibility tagger. Custom hmmEntityModel inputs remain future Text Analytics infrastructure. Sentence and part-of-speech details are ensured first, and recognized entities receive proper-noun part-of-speech tags.",
        "English and German rules classify person, location, organization, other, and non-entity. Existing Japanese/Korean metadata uses conservative token-class-aware fallback tags. Public MeCab-backed construction, entity models, and exact MathWorks statistical-model parity remain open.",
    ] },
    BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
        "The builtin retokenizes host strings and updates host tokenizedDocument metadata. It has no provider kernel or fusion path.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "add-entities", title: "Add entity details",
        program: "documents = tokenizedDocument(\"Mary uses MATLAB at MathWorks.\"); updated = addEntityDetails(documents); tdetails = tokenDetails(updated)",
        display_output: Some("tdetails.Entity includes person, organization, other, and non-entity tags"),
        compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(~isempty(tdetails.Entity));" },
    },
    BuiltinExample { id: "keep-tokens", title: "Keep existing tokens",
        program: "documents = tokenizedDocument(\"Mary visits New York.\"); updated = addEntityDetails(documents, \"RetokenizeMethod\", \"none\")",
        display_output: Some("updated has entity details without entity retokenization"),
        compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(updated), size(documents)));" },
    },
    BuiltinExample { id: "recompute-tags", title: "Recompute known tags",
        program: "documents = tokenizedDocument(\"Mary visits New York.\"); updated = addEntityDetails(documents, \"DiscardKnownValues\", true)",
        display_output: Some("updated has refreshed entity details"),
        compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(updated), size(documents)));" },
    },
];
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/text_analytics/entities.rs"),
};
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("addEntityDetails"), slug: Some("addEntityDetails"),
    summary: "Add named-entity tags to tokenized documents.",
    description: "addEntityDetails adds named-entity metadata to a RunMat tokenizedDocument compatibility object.",
    keywords: &["addEntityDetails", "tokenizedDocument", "text analytics", "entity", "named entity"],
    related: &["tokenDetails", "tokenizedDocument", "addPartOfSpeechDetails"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does addEntityDetails use a full entity model?", answer: "No. It uses deterministic English and German rules; exact MathWorks model parity remains open." },
        BuiltinDocumentationFaq { question: "Does addEntityDetails execute on the GPU?", answer: "No. It updates host tokenizedDocument metadata." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "tokenDetails", target: BuiltinDocumentationLinkTarget::Builtin("tokenDetails") },
        BuiltinDocumentationLink { label: "tokenizedDocument", target: BuiltinDocumentationLinkTarget::Builtin("tokenizedDocument") },
        BuiltinDocumentationLink { label: "addPartOfSpeechDetails", target: BuiltinDocumentationLinkTarget::Builtin("addPartOfSpeechDetails") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "entity controls, retokenization and labels", location: "builtins::strings::text_analytics::entities::tests" }],
        notes: &["Custom hmmEntityModel inputs, MeCab-backed construction, and exact statistical model parity remain open."],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
