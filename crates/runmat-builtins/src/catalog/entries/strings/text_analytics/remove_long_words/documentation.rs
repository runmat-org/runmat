use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("removeLongWords"),
 slug: Some("removeLongWords"),
 summary: "Remove long words from tokenized documents or bag-of-words models.",
 description: "`removeLongWords(documents, len)` removes tokens whose character length is greater than or equal to `len` from RunMat `tokenizedDocument` compatibility objects. `removeLongWords(bag, len)` removes matching vocabulary columns from RunMat `bagOfWords` objects.",
 keywords: &["removeLongWords", "tokenizedDocument", "bagOfWords", "text analytics", "filter"],
 related: &["tokenizedDocument", "bagOfWords", "removeShortWords", "removeWords"],
 sections: &[
BuiltinDocumentationSection { heading: "Supported forms", paragraphs: &["newDocuments = removeLongWords(documents, len)\nnewBag = removeLongWords(bag, len)", "`newDocuments` is a `tokenizedDocument` compatibility object. `newBag` is a `bagOfWords` compatibility object.", "`len` must be positive and integer-valued.", "Input objects must be RunMat Text Analytics compatibility objects from this module."] },
BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["`documents` must be a RunMat `tokenizedDocument` object created by `tokenizedDocument`.", "`bag` must be a RunMat `bagOfWords` object created by `bagOfWords`.", "`len` must be a positive integer scalar and accepts every integer class without a floating conversion.", "For tokenized documents, the result preserves `Shape`, `TokenizeMethod`, and `Language` metadata while recomputing document lengths and vocabulary.", "For bag-of-words models, the result preserves document row count and removes count columns for words with length greater than or equal to `len`."] },
BuiltinDocumentationSection { heading: "Execution and residency", paragraphs: &["`removeLongWords` filters host text-model objects. Automatic residency for a numeric control gathers transparently; explicit gpuArray host fallback is a declared RunMat extension.", "`removeLongWords` filters host text/model objects and has no provider kernel."] },
BuiltinDocumentationSection { heading: "Implementation boundary", paragraphs: &["The host path counts Unicode characters in each token and filters model vocabulary/count columns. It does not introduce sparse bag storage or a provider text-filter kernel."] },
BuiltinDocumentationSection { heading: "Finite host allocation", paragraphs: &["The current implementation requires an Unbounded host allocation policy. Under Managed finite allocation, a shared family guard rejects before gather, model cloning, mutation or filesystem work because complete text-model scratch and metadata ownership is not yet implemented. Authority is checked again after suspension and before publication, and failures are terminal rather than permission to replay. This refusal is an implementation frontier, not completion of finite resource support."] },
 ],
 examples: &[
BuiltinExample { id: "retained-1", title: "Filter Tokenized Documents", program: "documents = tokenizedDocument(\"An example of a short sentence\");\nnewDocuments = removeLongWords(documents, 7)", display_output: Some("`newDocuments` contains `An of a short`."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(newDocuments.Vocabulary, [\"An\" \"of\" \"a\" \"short\"])); assert(newDocuments.DocumentLengths == 4);" } },
BuiltinExample { id: "retained-2", title: "Filter A Bag Of Words", program: "documents = tokenizedDocument([\"an example of a short sentence\"; \"a second short sentence\"]);\nbag = bagOfWords(documents);\nnewBag = removeLongWords(bag, 7)", display_output: Some("`newBag.Vocabulary` excludes `example` and `sentence`."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(newBag.Vocabulary, [\"an\" \"of\" \"a\" \"short\" \"second\"])); assert(isequal(newBag.Counts, [1 1 1 1 0; 0 0 1 1 1]));" } },
 ],
 example_exemption: None,
 faqs: &[
BuiltinDocumentationFaq { question: "Does `removeLongWords` remove words longer than len or longer than or equal to len?", answer: "It removes words with character length greater than or equal to `len`, matching the MATLAB Text Analytics contract." },
BuiltinDocumentationFaq { question: "Does `removeLongWords` support raw strings?", answer: "No. Convert raw text with `tokenizedDocument` first, or build a `bagOfWords` model and filter that object." },
BuiltinDocumentationFaq { question: "Does `removeLongWords` execute on the GPU?", answer: "No. It filters host text/model metadata and has no `runmat-accelerate` provider path." },
 ],
 links: &[
BuiltinDocumentationLink { label: "tokenizedDocument", target: BuiltinDocumentationLinkTarget::Builtin("tokenizedDocument") },
BuiltinDocumentationLink { label: "bagOfWords", target: BuiltinDocumentationLinkTarget::Builtin("bagOfWords") },
BuiltinDocumentationLink { label: "removeShortWords", target: BuiltinDocumentationLinkTarget::Builtin("removeShortWords") },
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
