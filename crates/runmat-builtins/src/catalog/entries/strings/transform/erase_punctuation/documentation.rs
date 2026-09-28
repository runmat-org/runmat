use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Syntax and behavior", paragraphs: &[
        "newStr = erasePunctuation(str) removes Unicode punctuation (P*) and symbol (S*) characters from string arrays, character vectors or cell arrays of character vectors and preserves the documented container type.",
        "newDocuments = erasePunctuation(documents) processes punctuation and other token types by default while preserving complex tokens such as web addresses and email addresses. The TokenTypes name-value form applies the transform only to the selected built-in or custom token types.",
        "RunMat preserves document shape and explicit token-type details in its current scalar compatibility-object representation. First-class object-array identity remains separate infrastructure work.",
        "Integer and logical values are invalid in text, document and TokenTypes roles. Character matrices and nested or non-cellstr text cells are separately gated RunMat-only extensions.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
        "erasePunctuation is host text processing with no resident-text representation, provider hook or fusion route. Invalid resident numeric values reject before provider access or gathering.",
        "The runtime binding transforms text or tokenizedDocument without callbacks or external resources. Invalid options and malformed input may throw.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "legacy-text", title: "Erase punctuation from text", program: "newStr = erasePunctuation(\"it's one and/or two.\")", display_output: Some("newStr = \"its one andor two\""), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(newStr, \"its one andor two\"));" } },
    BuiltinExample { id: "legacy-documents", title: "Preserve complex document tokens by default", program: "documents = tokenizedDocument(\"Visit https://example.com.\");\nnewDocuments = erasePunctuation(documents)", display_output: Some("The web-address token remains intact while the sentence punctuation token is removed."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(newDocuments.Documents{1}, [\"Visit\" \"https://example.com\"]));" } },
];
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/compat.rs") };
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("erasePunctuation"), slug: Some("erasePunctuation"),
    summary: "Erase Unicode punctuation and symbol characters from text or selected token types in tokenized documents.",
    description: "erasePunctuation removes Unicode punctuation and symbol characters from documented text containers. For tokenized documents it transforms only the selected token types and removes tokens that become empty.",
    keywords: &["erasePunctuation", "punctuation", "Unicode", "tokenizedDocument", "text analytics"], related: &["eraseURLs", "tokenizedDocument"], sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Does erasePunctuation run on a GPU?", answer: "No. It is host text processing without a provider kernel; resident numeric values are invalid text and reject without gathering." }, BuiltinDocumentationFaq { question: "Are tokenizedDocument arrays represented exactly?", answer: "RunMat currently uses one scalar compatibility object carrying document cells and Shape metadata. First-class object-array identity remains a general runtime infrastructure gap." }],
    links: &[BuiltinDocumentationLink { label: "MathWorks erasePunctuation documentation", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/textanalytics/ref/tokenizeddocument.erasepunctuation.html") }, BuiltinDocumentationLink { label: "eraseURLs", target: BuiltinDocumentationLinkTarget::Builtin("eraseURLs") }, BuiltinDocumentationLink { label: "tokenizedDocument", target: BuiltinDocumentationLinkTarget::Builtin("tokenizedDocument") }, SOURCE], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "text and extension admission", location: "builtins::strings::transform::compat::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "token type transformation", location: "builtins::strings::text_analytics::documents::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "identity inference", location: "crates/runmat-builtins/src/catalog/entries/strings/transform/erase_punctuation/inference.rs" }], notes: &["Text and document examples assert their resulting values. Typed signatures and placement describe supported inputs and host execution."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
