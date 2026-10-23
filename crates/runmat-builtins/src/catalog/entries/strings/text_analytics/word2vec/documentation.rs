use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("word2vec"),
 slug: Some("word2vec"),
 summary: "Map words to embedding vectors.",
 description: "`word2vec(emb, words)` returns one embedding-vector row per requested word from a `wordEmbedding` object.",
 keywords: &["word2vec", "wordEmbedding", "text analytics", "vectors"],
 related: &["readWordEmbedding", "vec2word"],
 sections: &[
BuiltinDocumentationSection { heading: "Supported forms", paragraphs: &["M = word2vec(emb, words)\nM = word2vec(emb, words, 'IgnoreCase', true)", "`M` is a numeric matrix with one row per input word.", "Input words are flattened into lookup order.", "`IgnoreCase` must be a logical scalar."] },
BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["`emb` must be a `wordEmbedding` object, normally created by `readWordEmbedding`.", "`words` may be a string scalar, string array, character vector, character matrix, or cell array of scalar text.", "The output is an `n`-by-`emb.Dimension` numeric matrix, where `n` is the number of requested words after flattening the input text container.", "Words missing from the vocabulary return `NaN` rows.", "`word2vec(___, 'IgnoreCase', true)` performs case-insensitive vocabulary lookup and uses the first matching vocabulary entry.", "Words and option names must be host text, and `IgnoreCase` must be logical. Native integer and resident numeric inputs are rejected rather than converted to text."] },
BuiltinDocumentationSection { heading: "Execution and residency", paragraphs: &["`word2vec` looks up host-resident text/model metadata and returns a numeric matrix; it does not use provider kernels.", "`word2vec` reads host model metadata and returns a CPU double matrix. Numeric resident word inputs are rejected before provider access."] },
BuiltinDocumentationSection { heading: "Implementation boundary", paragraphs: &["Vectors are returned as host double rows from the compatibility model. Single model storage, subword synthesis and provider-resident lookup are not implemented by this owner."] },
BuiltinDocumentationSection { heading: "Finite host allocation", paragraphs: &["The current implementation requires an Unbounded host allocation policy. Under Managed finite allocation, a shared family guard rejects before gather, model cloning, mutation or filesystem work because complete text-model scratch and metadata ownership is not yet implemented. Authority is checked again after suspension and before publication, and failures are terminal rather than permission to replay. This refusal is an implementation frontier, not completion of finite resource support."] },
 ],
 examples: &[
BuiltinExample { id: "retained-1", title: "Look Up Word Vectors", program: "emb = readWordEmbedding('vectors.vec');\nM = word2vec(emb, [\"king\" \"queen\"])", display_output: Some("`M` has two rows and `emb.Dimension` columns."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "word2vec-1" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: &[BuiltinFilesystemEntry::File { relative_path: "vectors.vec", content: BuiltinFixtureContent::Utf8("3 2\nking 1 0\nqueen 0 1\nMATLAB 1 1\n") }] }), requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(M, [1 0; 0 1])); assert(isequal(size(M), [2 emb.Dimension]));" } },
BuiltinExample { id: "retained-2", title: "Ignore Case", program: "emb = readWordEmbedding(\"vectors.vec\");\nM = word2vec(emb, \"MATLAB\", \"IgnoreCase\", true)", display_output: Some("`M` uses a case-insensitive vocabulary lookup."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "word2vec-2" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: &[BuiltinFilesystemEntry::File { relative_path: "vectors.vec", content: BuiltinFixtureContent::Utf8("3 2\nking 1 0\nqueen 0 1\nMATLAB 1 1\n") }] }), requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(M, [1 1]));" } },
 ],
 example_exemption: None,
 faqs: &[
BuiltinDocumentationFaq { question: "What happens for missing words?", answer: "RunMat returns a row of `NaN` values for each missing vocabulary word." },
BuiltinDocumentationFaq { question: "Does `word2vec` execute on the GPU?", answer: "No. It is a host lookup into a text model object." },
 ],
 links: &[
BuiltinDocumentationLink { label: "readWordEmbedding", target: BuiltinDocumentationLinkTarget::Builtin("readWordEmbedding") },
BuiltinDocumentationLink { label: "vec2word", target: BuiltinDocumentationLinkTarget::Builtin("vec2word") },
 ],
 media: &[],
 evidence: BuiltinDocumentationEvidence { implementation: &[
BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/strings/text_analytics/embeddings.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/text_analytics/embeddings.rs") },
 ], verification: &[
BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "unit", location: "builtins::strings::text_analytics::embeddings::tests" },
 ], notes: &[] },
 introduced: None,
 status: Some(BuiltinDocumentationStatus::Partial),
};
