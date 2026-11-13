use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("get.p"), slug: Some("get.p"),
 summary: "Read the conventional p_backing field of a value object.",
 description: "Read the conventional p_backing field of a value object.",
 keywords: &["dependent property",
"classdef",
"value object"], related: &[],
 sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
"This method-only runtime compatibility handler supports value objects, not handle objects. It does not invoke user property accessors.",
"The backing value may be a nested integer, resident or foreign value. It is forwarded without conversion; get.p returns double zero if p_backing is absent. set.p returns the updated value object."
] },
BuiltinDocumentationSection { heading: "Execution and ownership", paragraphs: &[
"Class metadata or value-object backing fields are inspected on the host without provider access. Nested resident and foreign values are not materialized.",
"Native and interpreted execution use the same portable runtime implementation. Class registry changes invalidate hierarchy assumptions.",
"Direct distributed inputs are unsupported. Ordinary workers require transferable values, available class metadata and callable/service dependencies; resident and foreign members retain their owner and affinity requirements.",
"Shared runtime placement owns timing and transfer accounting; this identity adds no scheduler or hardware thresholds."
] },
BuiltinDocumentationSection { heading: "Documentation reconciliation", paragraphs: &[
"Legacy title, category, keywords, summary, description, behavior, examples, FAQs, links, source and test references are retained here or in typed declarations. Fusion flags and provider precisions are represented by the canonical placement contract; no feature is required. Display-only examples are explanatory; executable examples below carry value assertions."
] }],
 examples: &[BuiltinExample { id: "backing-field", title: "Value-object backing field", program: "obj = decomposition(eye(2)); result = feval('get.p',obj);", display_output: None,
 compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(result,0));" } }],
 example_exemption: None,
 faqs: &[],
 links: &[],
 media: &[],
 evidence: BuiltinDocumentationEvidence {
 implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/dependent_property.rs") }],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Semantic regression coverage", location: "builtins::introspection::dependent_property::tests" }], notes: &[],
 }, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
