use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Supported form", paragraphs: &[
        "updatedDocuments = addDependencyDetails(documents)",
        "`updatedDocuments` is a `tokenizedDocument` compatibility object.",
        "`documents` must be a RunMat `tokenizedDocument` object.",
        "No name-value options are supported for this MATLAB-compatible form.",
        "Use `tokenDetails(updatedDocuments)` to view the resulting `Head` and `Dependency` table columns.",
    ] },
    BuiltinDocumentationSection { heading: "Dependency metadata", paragraphs: &[
        "addDependencyDetails(documents) updates a RunMat tokenizedDocument object. It accepts no name-value options. tokenDetails then exposes Head, Dependency, and SentenceNumber columns; missing sentence-number details are created before dependency analysis.",
        "Head is the one-based index of a token's syntactic head in the same document; a root token uses zero. Dependency uses Universal Dependencies-style labels including root, nsubj, obj, det, amod, advmod, case, obl, aux, cc, mark, punct, nummod, and dep.",
        "RunMat uses deterministic English and German rules over its shared part-of-speech classifier. Existing Japanese or Korean language metadata receives conservative token-class-aware fallback dependencies. Exact MathWorks statistical-parser parity, MeCab-backed tokenization, and broader model parity remain open.",
    ] },
    BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
        "The builtin updates host tokenizedDocument metadata and has no provider kernel or fusion path. Numeric values are not document inputs.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "dependency-details", title: "Add dependency details",
    program: "documents = tokenizedDocument(\"The dogs chase cats.\"); updated = addDependencyDetails(documents); tdetails = tokenDetails(updated)",
    display_output: Some("tdetails includes Head, Dependency, and SentenceNumber columns"),
    compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions { source: "assert(~isempty(tdetails.Head)); assert(~isempty(tdetails.Dependency));" },
}];
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/text_analytics/dependencies.rs"),
};
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("addDependencyDetails"), slug: Some("addDependencyDetails"),
    summary: "Add dependency-head tags to tokenized documents.",
    description: "addDependencyDetails adds dependency-head metadata to a RunMat tokenizedDocument compatibility object.",
    keywords: &["addDependencyDetails", "tokenizedDocument", "text analytics", "dependency", "head"],
    related: &["tokenDetails", "tokenizedDocument", "addSentenceDetails"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does addDependencyDetails use a full parser model?", answer: "No. It uses deterministic compatibility rules; exact MathWorks statistical-parser parity remains open." },
        BuiltinDocumentationFaq { question: "Does addDependencyDetails execute on the GPU?", answer: "No. It updates host tokenizedDocument metadata." },
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
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "dependency labels and input validation", location: "builtins::strings::text_analytics::dependencies::tests" }],
        notes: &["Statistical parser and MeCab-backed language parity remain open."],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
