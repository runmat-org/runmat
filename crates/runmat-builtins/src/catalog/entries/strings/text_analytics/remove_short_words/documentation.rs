use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("removeShortWords"),
 slug: Some("removeShortWords"),
 summary: "Remove short words from tokenized documents or bag-of-words models.",
 description: "`removeShortWords(documents, len)` removes tokens whose character length is less than or equal to `len` from RunMat `tokenizedDocument` compatibility objects. `removeShortWords(bag, len)` removes matching vocabulary columns from RunMat `bagOfWords` objects.",
 keywords: &["removeShortWords", "tokenizedDocument", "bagOfWords", "text analytics", "filter"],
 related: &["tokenizedDocument", "bagOfWords", "erasePunctuation", "removeLongWords", "removeWords"],
 sections: &[
BuiltinDocumentationSection { heading: "Supported forms", paragraphs: &["newDocuments = removeShortWords(documents, len)\nnewBag = removeShortWords(bag, len)", "`newDocuments` is a `tokenizedDocument` compatibility object. `newBag` is a `bagOfWords` compatibility object.", "`len` must be positive and integer-valued.", "Input objects must be RunMat Text Analytics compatibility objects from this module."] },
BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["`documents` must be a RunMat `tokenizedDocument` object created by `tokenizedDocument`.", "`bag` must be a RunMat `bagOfWords` object created by `bagOfWords`.", "`len` must be a positive integer scalar and accepts every integer class without a floating conversion.", "For tokenized documents, the result preserves `Shape`, `TokenizeMethod`, and `Language` metadata while recomputing document lengths and vocabulary.", "For bag-of-words models, the result preserves document row count and removes count columns for words with length less than or equal to `len`.", "`removeStopWords`, `removeWords`, and `removeLongWords` are implemented separately for other tokenized-document filtering workflows. Table-backed custom token filters remain tracked by the Text Analytics umbrella."] },
BuiltinDocumentationSection { heading: "Execution and residency", paragraphs: &["`removeShortWords` filters host text-model objects. Automatic residency for a numeric control gathers transparently; explicit gpuArray host fallback is a declared RunMat extension.", "`removeShortWords` filters host text/model objects and has no provider kernel."] },
BuiltinDocumentationSection { heading: "Implementation boundary", paragraphs: &["The host path counts Unicode characters in each token and filters model vocabulary/count columns. It does not introduce sparse bag storage or a provider text-filter kernel."] },
BuiltinDocumentationSection { heading: "Finite host allocation", paragraphs: &["The current implementation requires an Unbounded host allocation policy. Under Managed finite allocation, a shared family guard rejects before gather, model cloning, mutation or filesystem work because complete text-model scratch and metadata ownership is not yet implemented. Authority is checked again after suspension and before publication, and failures are terminal rather than permission to replay. This refusal is an implementation frontier, not completion of finite resource support."] },
 ],
 examples: &[
BuiltinExample { id: "retained-1", title: "Filter Tokenized Documents", program: "documents = tokenizedDocument(\"a short document\");\nnewDocuments = removeShortWords(documents, 1)", display_output: Some("`newDocuments` contains `short` and `document`."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(newDocuments.Vocabulary, [\"short\" \"document\"])); assert(newDocuments.DocumentLengths == 2);" } },
BuiltinExample { id: "retained-2", title: "Filter A Bag Of Words", program: "documents = tokenizedDocument([\"a beta\"; \"an gamma\"]);\nbag = bagOfWords(documents);\nnewBag = removeShortWords(bag, 2)", display_output: Some("`newBag.Vocabulary` contains `beta` and `gamma`."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(newBag.Vocabulary, [\"beta\" \"gamma\"])); assert(isequal(newBag.Counts, [1 0; 0 1]));" } },
 ],
 example_exemption: None,
 faqs: &[
BuiltinDocumentationFaq { question: "Does `removeShortWords` remove words shorter than len or shorter than or equal to len?", answer: "It removes words with character length less than or equal to `len`, matching the MATLAB Text Analytics contract." },
BuiltinDocumentationFaq { question: "Does `removeShortWords` support raw strings?", answer: "No. Convert raw text with `tokenizedDocument` first, or build a `bagOfWords` model and filter that object." },
BuiltinDocumentationFaq { question: "Does `removeShortWords` execute on the GPU?", answer: "No. It filters host text/model metadata and has no `runmat-accelerate` provider path." },
 ],
 links: &[
BuiltinDocumentationLink { label: "tokenizedDocument", target: BuiltinDocumentationLinkTarget::Builtin("tokenizedDocument") },
BuiltinDocumentationLink { label: "bagOfWords", target: BuiltinDocumentationLinkTarget::Builtin("bagOfWords") },
BuiltinDocumentationLink { label: "erasePunctuation", target: BuiltinDocumentationLinkTarget::Builtin("erasePunctuation") },
BuiltinDocumentationLink { label: "removeLongWords", target: BuiltinDocumentationLinkTarget::Builtin("removeLongWords") },
BuiltinDocumentationLink { label: "removeWords", target: BuiltinDocumentationLinkTarget::Builtin("removeWords") },
 ],
 media: &[],
 evidence: BuiltinDocumentationEvidence { implementation: &[
BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/strings/text_analytics/documents.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/text_analytics/documents.rs") },
 ], verification: &[
BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "unit", location: "builtins::strings::text_analytics::documents::tests" },
BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "integration", location: "crates/runmat-vm/tests/integer_label_and_text_control_semantics.rs" },
 ], notes: &[] },
 introduced: None,
 status: Some(BuiltinDocumentationStatus::Partial),
};
