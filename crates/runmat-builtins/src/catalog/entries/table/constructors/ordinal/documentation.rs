use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("ordinal"), slug: Some("ordinal"),
 summary: "Create an ordinal categorical array object.", description: "`ordinal(X)` creates an ordered compatibility object from the sorted unique levels of `X`. Labels, explicit levels, and numeric bin edges follow the public ordinal argument order.",
 keywords: &["ordinal", "categorical", "categories", "statistics"], related: &[],
 sections: &[
BuiltinDocumentationSection {
 heading: "Behavior",
 paragraphs: &[
"String, char, cellstr, numeric, and logical inputs are accepted through the same label conversion path as `categorical`.",
"`ordinal(X)` derives sorted unique levels from non-missing values and creates default labels.",
"`ordinal(X, labels)` assigns the supplied text labels to the sorted unique levels; duplicate labels merge levels.",
"`ordinal(X, labels, levels)` uses the explicit comparable level vector and marks source values absent from it as undefined.",
"`ordinal(X, labels, [], edges)` bins numeric source values into left-closed, right-open intervals, with the final edge included in the last bin.",
"Integer source values, explicit levels, and edges use exact native comparison across all eight classes. Default numeric display labels retain the established five-significant-digit formatting and collision rejection.",
"The returned object has class `categorical`, stores one-based category codes, and has its `Ordinal` metadata flag set to true.",
"Equality and relational operators compare category order; categorical-to-categorical comparisons require matching category lists.",
"`min` and `max` reduce ordinal categorical arrays by category order and return categorical results; undefined elements follow the numeric `NaN` behavior of `includenan` and `omitnan`, and multi-output forms return numeric indices.",
],
},
BuiltinDocumentationSection {
 heading: "Residency and fusion",
 paragraphs: &[
"Ordinal compatibility objects are host metadata. Automatic residency gathers transparently; explicit gpuArray host construction is a declared RunMat extension.",
"The canonical default binding retains arguments across provider awaits and shares categorical metadata construction. It may suspend, is a host/fusion boundary, and is portable through the Runtime on native and Wasm; no GPU kernel or distributed implementation is advertised.",
"Finite Managed allocation is refused before any gather or raw category/bin workspace. Exact quoted label/code/metadata ownership remains L25 work; the refusal is not a claim of finite allocation support.",
],
},
BuiltinDocumentationSection {
 heading: "Syntax and ordering",
 paragraphs: &[
"The output is a categorical object, not a distinct ordinal class. Its Ordinal and Protected flags are true. Character rows become observations; other supported source arrays retain their full code shape and cell labels use column-major order.",
"The two-argument form assigns labels to inferred unique levels, so the label count must match that level count. To retain an unused level, supply it explicitly as the third argument. The original two-observation/three-label example is therefore retained as an expected-error witness; the following corrected example supplies explicit levels.",
"For the edge form, labels number one fewer than edges, edges must be an increasing vector, and the third argument must be empty. Out-of-range or missing source values remain undefined. Invalid level/label counts and duplicate mapping levels reject.",
],
}
 ],
 examples: &[
BuiltinExample {
 id: "reject-mismatched-label-count", title: "Create an ordered categorical array", program: "O = ordinal({'medium'; 'low'}, {'low','medium','high'})",
 display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::ExpectedError { identifier: "RunMat:table:InvalidArgument" },
},
BuiltinExample {
 id: "explicit-unused-level", title: "Retain an explicit unused category", program: "O = ordinal({'medium'; 'low'}, {'low','medium','high'}, {'low','medium','high'});",
 display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions { source: "assert(isordinal(O));\nassert(isequal(O == 'low', [false; true]));\nassert(isequal(O > 'low', [true; false]));" },
}
 ],
 example_exemption: None,
 faqs: &[BuiltinDocumentationFaq { question: "How is `ordinal` different from `categorical`?", answer: "`ordinal` always sets the categorical `Ordinal` flag to true, so category order is meaningful for compatibility code that checks ordinal categorical metadata." }],
 links: &[
BuiltinDocumentationLink { label: "categorical", target: BuiltinDocumentationLinkTarget::Builtin("categorical") },
BuiltinDocumentationLink { label: "iscategorical", target: BuiltinDocumentationLinkTarget::Builtin("iscategorical") },
BuiltinDocumentationLink { label: "isordinal", target: BuiltinDocumentationLinkTarget::Builtin("isordinal") }
 ],
 media: &[],
 evidence: BuiltinDocumentationEvidence {
 implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/table/builtins/constructors.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/builtins/constructors.rs") }],
 verification: &[
BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "builtins::table::tests::ordinal_constructor_sets_ordered_categorical_semantics", location: "crates/runmat-runtime/src/builtins/table/tests.rs" },
BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "crates/runmat-vm/tests/integer_label_and_text_control_semantics.rs", location: "crates/runmat-vm/tests/integer_label_and_text_control_semantics.rs" }
 ],
 notes: &["Source examples have independent assertions or a typed expected-error witness; display transcripts are not fabricated.", "Legacy GPU/fusion metadata is normalized to the typed host/fusion boundary contract. No optional feature is introduced; finite allocation remains an explicit L25 obligation."],
 },
 introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
