use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("isVocabularyWord"),
 slug: Some("isVocabularyWord"),
 summary: "Test whether words are in a wordEmbedding or wordEncoding vocabulary.",
 description: "`isVocabularyWord(emb, words)` and `isVocabularyWord(enc, words)` return logical masks for vocabulary membership.",
 keywords: &["isVocabularyWord", "wordEmbedding", "wordEncoding", "text analytics", "vocabulary"],
 related: &["wordEncoding", "word2ind", "ind2word", "fastTextWordEmbedding"],
 sections: &[
BuiltinDocumentationSection { heading: "Supported forms", paragraphs: &["tf = isVocabularyWord(emb, words)\ntf = isVocabularyWord(enc, words)\ntf = isVocabularyWord(___, 'IgnoreCase', true)", "`tf` is a logical array.", "`IgnoreCase` must be a logical scalar."] },
BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["The first input must be a RunMat `wordEmbedding` or `wordEncoding` compatibility object.", "`words` may be a string scalar, string array, character vector, character matrix, or cell array of scalar text.", "The output logical array preserves the input word container shape for string arrays and cell arrays. A character vector returns a scalar 1-by-1 logical array; a character matrix returns one row per input row.", "`isVocabularyWord(___, 'IgnoreCase', true)` performs case-insensitive vocabulary lookup."] },
BuiltinDocumentationSection { heading: "Execution and residency", paragraphs: &["`isVocabularyWord` is a host vocabulary lookup and has no provider kernel.", "`isVocabularyWord` reads host text/model metadata and returns a CPU logical array."] },
BuiltinDocumentationSection { heading: "Implementation boundary", paragraphs: &["Each call builds the existing host vocabulary lookup and returns a logical mask with word-input geometry. Numeric text coercion and provider lookup are not supported."] },
BuiltinDocumentationSection { heading: "Finite host allocation", paragraphs: &["The current implementation requires an Unbounded host allocation policy. Under Managed finite allocation, a shared family guard rejects before gather, model cloning, mutation or filesystem work because complete text-model scratch and metadata ownership is not yet implemented. Authority is checked again after suspension and before publication, and failures are terminal rather than permission to replay. This refusal is an implementation frontier, not completion of finite resource support."] },
 ],
 examples: &[
BuiltinExample { id: "retained-1", title: "Test Membership", program: "enc = wordEncoding([\"I\" \"love\" \"RunMat\"]);\ntf = isVocabularyWord(enc, [\"RunMat\" \"MATLAB\"])", display_output: Some("`tf` is `[true false]`."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(islogical(tf)); assert(isequal(tf, [true false]));" } },
 ],
 example_exemption: None,
 faqs: &[
BuiltinDocumentationFaq { question: "Does this work for both embeddings and encodings?", answer: "Yes. RunMat supports `wordEmbedding` objects from `fastTextWordEmbedding`, `readWordEmbedding`, or `trainWordEmbedding`, and `wordEncoding` objects from `wordEncoding`." },
BuiltinDocumentationFaq { question: "Does `isVocabularyWord` execute on the GPU?", answer: "No. It is a host vocabulary lookup." },
 ],
 links: &[
BuiltinDocumentationLink { label: "wordEncoding", target: BuiltinDocumentationLinkTarget::Builtin("wordEncoding") },
BuiltinDocumentationLink { label: "word2ind", target: BuiltinDocumentationLinkTarget::Builtin("word2ind") },
BuiltinDocumentationLink { label: "ind2word", target: BuiltinDocumentationLinkTarget::Builtin("ind2word") },
BuiltinDocumentationLink { label: "fastTextWordEmbedding", target: BuiltinDocumentationLinkTarget::Builtin("fastTextWordEmbedding") },
 ],
 media: &[],
 evidence: BuiltinDocumentationEvidence { implementation: &[
BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/strings/text_analytics/encoding.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/text_analytics/encoding.rs") },
 ], verification: &[
BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "unit", location: "builtins::strings::text_analytics::encoding::tests" },
 ], notes: &[] },
 introduced: None,
 status: Some(BuiltinDocumentationStatus::Partial),
};
