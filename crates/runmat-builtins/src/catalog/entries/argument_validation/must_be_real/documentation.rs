use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog, title: Some("mustBeReal"), slug: Some("mustBeReal"), summary: "Validate that numeric values are real-valued.", description: "`mustBeReal` is available as a callable MATLAB argument validator. It returns no user-facing value when validation passes and throws a structured RunMat error when validation fails. Validators with literal/static payloads are also encoded in MATLAB `arguments` blocks; dynamic validator payload expressions should call the validator directly until runtime-evaluated arguments-block payloads are implemented.", keywords: &["mustBeReal",
"argument validation",
"arguments block",
"MATLAB compatibility"], related: &[],
 sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &["Callable validator forms enforce exact supported arity, throw on validation failure, and suppress automatic output on success.",
"All eight integer classes are real by construction and pass from authoritative class/storage metadata without floating conversion; empty values pass, and complex floating or paired-integer values are checked from their imaginary components.",
"Resident real integer arrays pass without payload access, while resident complex floating arrays gather through their exact owner only when their imaginary values must be inspected.",
"Arguments-block validation uses the same shared runtime predicates for validators that RunMat can encode statically.",
"Numeric validators check every element without materializing a cloned numeric array, including sparse stored values and implicit sparse zeros where relevant.",
"Text validators accept string scalars/arrays, character row vectors, and compatible cell arrays of text where MATLAB commonly permits them."] },
BuiltinDocumentationSection { heading: "Residency and provider behavior", paragraphs: &["Validators run on host values and metadata; they throw on validation failure."] },
BuiltinDocumentationSection { heading: "Execution and service boundaries", paragraphs: &["This validator has zero public outputs. Statements and arguments-block checks validate normally; assigning a result rejects before provider or filesystem work.",
"Content-dependent resident checks use the existing exact-owner, typed, non-destructive download service. Metadata-decidable integer/logical cases avoid downloads. No validator device kernels or automatic uploads are used.",
"NaN and missingness checks traverse owned cells, not object references. The mustBeNonNan explicit-resident extension applies to nested resident values before downloads; mustBeNonmissing has no such extension."] }],
 examples: &[BuiltinExample { id:"guard", title:"Using the validator as a normal callable guard", program:"A = uint64(7);\nmustBeReal(A);\npassed = true;", display_output: Some("No output when validation passes; an error is thrown when validation fails."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source:"assert(passed);" } }], example_exemption: None,
 faqs: &[BuiltinDocumentationFaq { question:"Does this validator work in an `arguments` block?", answer:"Yes for literal/static validator payloads that RunMat can encode at compile time. Dynamic validator payload expressions should use the callable form for now." }], links: &[BuiltinDocumentationLink { label:"`arguments` blocks", target:BuiltinDocumentationLinkTarget::Documentation("../language/arguments") }], media: &[],
 evidence: BuiltinDocumentationEvidence {
 implementation: &[BuiltinDocumentationLink {
 label:"Implementation",
 target:BuiltinDocumentationLinkTarget::Source("crates/runmat-runtime/src/builtins/common/validation/must_be_real.rs") }],
 verification: &[BuiltinEvidenceReference {
 kind:BuiltinEvidenceKind::UnitTest,
 label:"Validator domain tests",
 location:"builtins::common::validation::tests" }],
 notes: &["Retained unit reference: crates/runmat-runtime/src/builtins/common/validation.rs table-style pass/fail tests",
"Retained integration reference: Representative callable and arguments-block coverage in crates/runmat-vm/tests/functions.rs; not every validator has a dedicated VM case.",
"Legacy GPU/fusion flags are reconciled into typed domain-owned placement; no validator GPU kernel is claimed. Integrated execution remains required."] },
 introduced: None,
 status:Some(BuiltinDocumentationStatus::Stable),

};
