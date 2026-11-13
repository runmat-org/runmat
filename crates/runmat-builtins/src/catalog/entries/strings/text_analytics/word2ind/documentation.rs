use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("word2ind"),
 slug: Some("word2ind"),
 summary: "Map words to indices in a wordEncoding object.",
 description: "`word2ind(enc, words)` returns vocabulary indices for words in a `wordEncoding` object.",
 keywords: &["word2ind", "wordEncoding", "text analytics", "indices"],
 related: &["wordEncoding", "ind2word", "isVocabularyWord"],
 sections: &[
BuiltinDocumentationSection { heading: "Supported forms", paragraphs: &["M = word2ind(enc, words)\nM = word2ind(enc, words, 'IgnoreCase', true)", "`M` is a numeric tensor of 1-based indices or NaN values.", "`IgnoreCase` must be a logical scalar."] },
BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["`enc` must be a `wordEncoding` object.", "`words` may be a string scalar, string array, character vector, character matrix, or cell array of scalar text.", "The output numeric tensor preserves the input word container shape for string arrays and cell arrays. A character vector returns a scalar 1-by-1 tensor; a character matrix returns one row per input row.", "Indices are 1-based. Words missing from the vocabulary return `NaN`.", "`word2ind(___, 'IgnoreCase', true)` performs case-insensitive vocabulary lookup and uses the first matching vocabulary entry.", "The input surface is textual: native integer arrays and resident numeric arrays are rejected rather than converted to words. The returned indices are double values, with `NaN` representing missing words."] },
BuiltinDocumentationSection { heading: "Execution and residency", paragraphs: &["`word2ind` is a host vocabulary lookup and has no provider kernel.", "`word2ind` reads host text and model metadata and returns a CPU double tensor. Numeric resident inputs are rejected before provider access because they are not valid word inputs."] },
BuiltinDocumentationSection { heading: "Implementation boundary", paragraphs: &["Each lookup builds a host vocabulary map and returns host double indices, retaining NaN for unknown words. Numeric text coercion and provider lookup are not supported."] },
BuiltinDocumentationSection { heading: "Finite host allocation", paragraphs: &["The current implementation requires an Unbounded host allocation policy. Under Managed finite allocation, a shared family guard rejects before gather, model cloning, mutation or filesystem work because complete text-model scratch and metadata ownership is not yet implemented. Authority is checked again after suspension and before publication, and failures are terminal rather than permission to replay. This refusal is an implementation frontier, not completion of finite resource support."] },
 ],
 examples: &[
BuiltinExample { id: "retained-1", title: "Map Words To Indices", program: "enc = wordEncoding([\"rose\" \"love\" \"beauty\"]);\nidx = word2ind(enc, [\"rose\" \"missing\"])", display_output: Some("`idx` is `[1 NaN]`."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(idx), [1 2])); assert(idx(1) == 1); assert(isnan(idx(2)));" } },
 ],
 example_exemption: None,
 faqs: &[
BuiltinDocumentationFaq { question: "What happens for missing words?", answer: "RunMat returns `NaN` for words outside the encoding vocabulary." },
BuiltinDocumentationFaq { question: "Does `word2ind` execute on the GPU?", answer: "No. It is a host text lookup." },
 ],
 links: &[
BuiltinDocumentationLink { label: "wordEncoding", target: BuiltinDocumentationLinkTarget::Builtin("wordEncoding") },
BuiltinDocumentationLink { label: "ind2word", target: BuiltinDocumentationLinkTarget::Builtin("ind2word") },
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
