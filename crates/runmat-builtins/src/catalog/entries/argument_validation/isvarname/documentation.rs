use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog, title: Some("isvarname"), slug: Some("isvarname"),
 summary: "Return true when text is a valid MATLAB variable name.", description: "`isvarname(S)` returns a logical scalar indicating whether text is a valid MATLAB variable name: nonempty, starts with a letter, is at most `namelengthmax` characters, contains only letters, digits, and underscores after the first character, and is not a MATLAB keyword.", keywords: &["isvarname", "name-value arguments", "MATLAB compatibility"],
 related: &[], sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Returns a normal logical value and does not mutate inputs.",
"Accepts MATLAB text representations supported by RunMat.",
"Rejects names that start with underscores or digits, MATLAB keywords such as `for` and `end`, and names longer than `namelengthmax` characters."] },
BuiltinDocumentationSection { heading: "Residency and provider behavior", paragraphs: &["Host metadata/text helper."] },
BuiltinDocumentationSection { heading: "Execution contract", paragraphs: &["The runtime binding is portable across interpreter, native/JIT/AOT and WASM routes. Metadata and text checks do not upload or gather operands. Invalid output counts and controls reject before provider payload access.",
"Zero requested outputs still execute the operation; at most one result is available.",
"The operation has no callbacks, filesystem, randomness, workspace mutation or foreign-runtime calls. Distributed values are inspected only as opaque handles, never materialized or validated partition-by-partition."] }],
 examples: &[BuiltinExample {
 id: "example-1", title: "Basic use", program: "tf = isvarname('valid_name');",
 display_output: Some("Result depends on input value."), compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions { source: "assert(tf); assert(~isvarname('1invalid'));" },
}], example_exemption: None,
 faqs: &[],
 links: &[],
 media: &[],
 evidence: BuiltinDocumentationEvidence {
  implementation: &[BuiltinDocumentationLink {
   label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("crates/runmat-runtime/src/builtins/common/validation/isvarname.rs"),
  }],
  verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Validator domain tests", location: "builtins::common::validation::tests" }],
  notes: &["Retained unit reference: crates/runmat-runtime/src/builtins/common/validation.rs varname positive and negative tests",
"Retained integration reference: crates/runmat-vm/tests/functions.rs::argument_validation_helpers_are_callable_builtins",
   "Legacy GPU/fusion declarations are reconciled into typed placement; no device kernel or payload preparation is claimed. Exact-source integrated runtime, native and documentation execution remains required.",
  ],
 },
 introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
