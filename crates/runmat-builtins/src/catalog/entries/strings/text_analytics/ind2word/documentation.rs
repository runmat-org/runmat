use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("ind2word"),
 slug: Some("ind2word"),
 summary: "Map wordEncoding indices back to words.",
 description: "`ind2word(enc, M)` returns the vocabulary words corresponding to 1-based indices in `M`.",
 keywords: &["ind2word", "wordEncoding", "text analytics", "vocabulary"],
 related: &["wordEncoding", "word2ind", "isVocabularyWord"],
 sections: &[
BuiltinDocumentationSection { heading: "Supported form", paragraphs: &["words = ind2word(enc, M)", "`words` is a string array.", "`M` must contain positive integer indices within `enc.NumWords`."] },
BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["`enc` must be a RunMat `wordEncoding` compatibility object.", "`M` may be a numeric scalar or numeric tensor of positive integer indices.", "MATLAB-compatible mode accepts host double index vectors. Native typed-integer vectors, nonvector index arrays, and resident indices are separate `runmat` mode extensions; admitted integer indices are read exactly.", "The output is a string array with the same shape as `M`.", "Indices must be finite positive integers within the vocabulary size. Invalid, NaN, fractional, zero, negative, or out-of-range indices are rejected with a controlled error."] },
BuiltinDocumentationSection { heading: "Execution and residency", paragraphs: &["`ind2word` is a host vocabulary lookup and has no provider kernel.", "`ind2word` reads host text/model metadata and returns a CPU string array."] },
BuiltinDocumentationSection { heading: "Implementation boundary", paragraphs: &["Indices are validated from their authoritative numeric representation before host vocabulary lookup. Typed integer, nonvector, and resident forms retain their independent extension gates; output strings remain host-owned."] },
BuiltinDocumentationSection { heading: "Finite host allocation", paragraphs: &["The current implementation requires an Unbounded host allocation policy. Under Managed finite allocation, a shared family guard rejects before gather, model cloning, mutation or filesystem work because complete text-model scratch and metadata ownership is not yet implemented. Authority is checked again after suspension and before publication, and failures are terminal rather than permission to replay. This refusal is an implementation frontier, not completion of finite resource support."] },
 ],
 examples: &[
BuiltinExample { id: "retained-1", title: "Map Indices To Words", program: "enc = wordEncoding([\"fairest\" \"desire\" \"thereby\"]);\nwords = ind2word(enc, [1 3])", display_output: Some("`words` is `[\"fairest\" \"thereby\"]`."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(words, [\"fairest\" \"thereby\"]));" } },
 ],
 example_exemption: None,
 faqs: &[
BuiltinDocumentationFaq { question: "Does `ind2word` return missing strings for invalid indices?", answer: "No. The documented input is positive integer indices, so RunMat rejects invalid or out-of-range indices." },
BuiltinDocumentationFaq { question: "Does `ind2word` execute on the GPU?", answer: "No. It is a host metadata lookup." },
 ],
 links: &[
BuiltinDocumentationLink { label: "wordEncoding", target: BuiltinDocumentationLinkTarget::Builtin("wordEncoding") },
BuiltinDocumentationLink { label: "word2ind", target: BuiltinDocumentationLinkTarget::Builtin("word2ind") },
BuiltinDocumentationLink { label: "isVocabularyWord", target: BuiltinDocumentationLinkTarget::Builtin("isVocabularyWord") },
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
