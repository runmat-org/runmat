use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("superclasses"), slug: Some("superclasses"),
 summary: "Return visible superclass names for a class or object.",
 description: "`superclasses(objOrClass)` returns the visible superclass names for a registered class or object as a cell row of character vectors.",
 keywords: &["superclasses",
"classdef",
"inheritance",
"object",
"introspection"], related: &[],
 sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
"Class-name character vectors, string scalars, and 1x1 string arrays are interpreted as class names.",
"Object and handle-object inputs are resolved through their runtime class names.",
"ClassRef inputs query the referenced class.",
"Explicit gpuArray inputs query the `gpuArray` class metadata, while internal automatic residency queries the underlying host class without gathering provider buffers.",
"Superclass names are returned nearest-parent first, then continuing toward the root.",
"Classes with no registered visible superclasses return a 1x0 cell array.",
"RunMat currently models a single parent link per class. Multiple inheritance and hidden-superclass filtering are not represented until the broader metadata object model lands.",
"Unknown class names raise `RunMat:SuperclassesClassUnknown`; nonobject, nontext inputs raise `RunMat:SuperclassesClassInvalid`."
] },
BuiltinDocumentationSection { heading: "Execution and ownership", paragraphs: &[
"Class metadata or value-object backing fields are inspected on the host without provider access. Nested resident and foreign values are not materialized.",
"Native and interpreted execution use the same portable runtime implementation. Class registry changes invalidate hierarchy assumptions.",
"Direct distributed inputs are unsupported. Ordinary workers require transferable values, available class metadata and callable/service dependencies; resident and foreign members retain their owner and affinity requirements.",
"Shared runtime placement owns timing and transfer accounting; this identity adds no scheduler or hardware thresholds."
] },
BuiltinDocumentationSection { heading: "GPU behavior", paragraphs: &[
"Metadata-only class hierarchy query. Explicit gpuArray values query wrapper metadata; internal automatic residency queries underlying host-class metadata. Inputs stay resident."
] },
BuiltinDocumentationSection { heading: "Additional usage", paragraphs: &[
"Get superclass names from an object\nobj = Circle();\ns = superclasses(obj)\nExpected display: s = {'Shape'}",
"Get superclass names from a class name\ns = superclasses('Circle')\nExpected display: s = {'Shape'}",
"Class with no superclass\ns = superclasses('Point')\nExpected display: s = 1x0 cell array"
] },
BuiltinDocumentationSection { heading: "Documentation reconciliation", paragraphs: &[
"Legacy title, category, keywords, summary, description, behavior, examples, FAQs, links, source and test references are retained here or in typed declarations. Fusion flags and provider precisions are represented by the canonical placement contract; no feature is required. Display-only examples are explanatory; executable examples below carry value assertions."
] }],
 examples: &[BuiltinExample { id: "leaf", title: "leaf",
 program: "s = superclasses('uint64');", display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions { source: "assert(iscell(s)); assert(isequal(size(s),[1 0]));" } }],
 example_exemption: None,
 faqs: &[BuiltinDocumentationFaq { question: "Does `superclasses` return strings?", answer: "No. MATLAB returns a cell array of character vectors, and RunMat follows that shape." },
BuiltinDocumentationFaq { question: "Does it inspect full metadata objects?", answer: "It uses the current RunMat class registry parent chain. Full `matlab.metadata.Class` object lists and hidden-class filtering are still tracked separately." }],
 links: &[BuiltinDocumentationLink { label: "metaclass", target: BuiltinDocumentationLinkTarget::Documentation("./metaclass") },
BuiltinDocumentationLink { label: "class", target: BuiltinDocumentationLinkTarget::Documentation("./class") },
BuiltinDocumentationLink { label: "ismethod", target: BuiltinDocumentationLinkTarget::Documentation("./ismethod") }],
 media: &[],
 evidence: BuiltinDocumentationEvidence {
 implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/superclasses.rs") }],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Semantic regression coverage", location: "builtins::introspection::superclasses::tests" },
BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Semantic regression coverage", location: "crates/runmat-vm/tests/functions.rs::superclasses_script_surface_reports_registered_parent_chain" }], notes: &[],
 }, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
