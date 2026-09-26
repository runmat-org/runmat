use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog, title: Some("namedargs2cell"), slug: Some("namedargs2cell"),
 summary: "Convert a scalar name-value struct into an alternating name/value cell row.", description: "`namedargs2cell(S)` preserves scalar struct field order and returns `{name1, value1, name2, value2, ...}` for name-value forwarding patterns.", keywords: &["namedargs2cell", "name-value arguments", "MATLAB compatibility"],
 related: &[], sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["The input must be a scalar structure; top-level integer and resident numeric inputs reject without conversion or provider access.",
"Field names retain structure insertion order and field values are moved unchanged into alternating name/value cells.",
"All eight integer classes stored in fields retain their exact class and payload, including wide `int64` and `uint64` values."] },
BuiltinDocumentationSection { heading: "Residency and provider behavior", paragraphs: &["Host scalar-structure transformation; top-level resident numeric values reject without provider access."] },
BuiltinDocumentationSection { heading: "Execution contract", paragraphs: &["The runtime binding is portable across interpreter, native/JIT/AOT and WASM routes. Metadata and text checks do not upload or gather operands. Invalid output counts and controls reject before provider payload access.",
"Zero requested outputs still execute the operation; at most one result is available.",
"The operation has no callbacks, filesystem, randomness, workspace mutation or foreign-runtime calls. Distributed values are inspected only as opaque handles, never materialized or validated partition-by-partition."] },
BuiltinDocumentationSection { heading: "Ownership", paragraphs: &["Field values, including nested resident handles, move unchanged into the cell row. Names follow runtime insertion order; no sorted field-map order is promised by inference."] }],
 examples: &[BuiltinExample {
 id: "example-1", title: "Basic use", program: "C = namedargs2cell(struct('width',uint64(7),'label',\"x\"));",
 display_output: Some("Result depends on input value."), compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(C),[1 4])); assert(strcmp(C{1},'width')); assert(isa(C{2},'uint64')); assert(C{2} == uint64(7)); assert(strcmp(C{3},'label')); assert(strcmp(C{4},'x'));" },
}], example_exemption: None,
 faqs: &[],
 links: &[],
 media: &[],
 evidence: BuiltinDocumentationEvidence {
  implementation: &[BuiltinDocumentationLink {
   label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("crates/runmat-runtime/src/builtins/common/validation/namedargs2cell.rs"),
  }],
  verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Validator domain tests", location: "builtins::common::validation::tests" }],
  notes: &["Retained unit reference: builtins::common::validation::tests::namedargs2cell_rejects_top_level_integers_and_preserves_integer_fields_exactly",
"Retained integration reference: crates/runmat-vm/tests/integer_datetime_metadata_and_nonzero_semantics.rs",
   "Legacy GPU/fusion declarations are reconciled into typed placement; no device kernel or payload preparation is claimed. Exact-source integrated runtime, native and documentation execution remains required.",
  ],
 },
 introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
