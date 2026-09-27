use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("subsasgn"), slug: Some("subsasgn"),
 summary: "Assign through a standard substruct path on a mutable value.",
 description: "`subsasgn(A, S, rhs)` assigns through the nonempty structure-array path `S`, conventionally produced by `substruct`. Ordinary arrays, cells, strings, and structures use canonical default assignment; class objects may resolve a class-specific `subsasgn` implementation. Field, parenthesis, and brace steps remain ordered, and each overload receives the complete remaining path.",
 keywords: &["subsasgn",
"object assignment",
"overloaded indexing",
"dot assignment"], related: &[],
 sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
"The receiver may be any value that supports the requested default or overloaded assignment path.",
"Each element of `S` has a `type` field (`'.'`, `'()'`, or `'{}'`) and a corresponding `subs` field.",
"A multi-element `S` preserves the complete chained assignment path in source order.",
"Class objects with a `subsasgn` method receive the complete remaining path; values without an overload use canonical default assignment.",
"Integer selectors and assigned values retain their native class, exact value, shape, and resident handle during dispatch. The class method owns assignment conversion, overflow, and returned-receiver semantics.",
"Prefer normal assignment syntax in user code; direct calls to `subsasgn` are mainly useful for advanced dispatch or testing."
] },
BuiltinDocumentationSection { heading: "Execution and ownership", paragraphs: &[
"Default indexing uses the shared object protocol. Class methods may throw, suspend, mutate handles, access ambient services or invoke arbitrary code. Unknown overload effects and capabilities remain conservative.",
"Native and interpreted execution share the portable runtime protocol and process-stack reentry. Object/class registry dependencies and callback captures follow shared GC rooting and affinity rules; retries cannot replay overload effects.",
"Direct distributed inputs are unsupported. Ordinary workers require transferable values, available class metadata and callable/service dependencies; resident and foreign members retain their owner and affinity requirements.",
"Shared runtime placement owns timing and transfer accounting; this identity adds no scheduler or hardware thresholds."
] },
BuiltinDocumentationSection { heading: "GPU behavior", paragraphs: &[
"Method resolution executes on the host, but selector payloads and right-hand-side values are forwarded to the class implementation without numeric conversion or an implicit residency change.",
"The shared default-indexing service owns numeric provider operations and materialization. Class overloads determine their own provider work; dispatch does not gather nested values.",
"`subsasgn` is host-side object dispatch and does not itself manage GPU residency."
] },
BuiltinDocumentationSection { heading: "Additional usage", paragraphs: &[
"Change a datetime display format explicitly\nt = datetime(2024, 4, 9);\nS = substruct('.', 'Format');\nt = subsasgn(t, S, \"yyyy-MM-dd\");\ndisp(t)\nExpected display: 2024-04-09",
"Replace one element of a datetime array explicitly\nt = datetime([2024 2025], [1 6], [15 20]);\nS = substruct('()', {2});\nt = subsasgn(t, S, datetime(2030, 1, 1));\ndisp(t)\nExpected display: 15-Jan-2024 00:00:00\n01-Jan-2030 00:00:00"
] },
BuiltinDocumentationSection { heading: "Documentation reconciliation", paragraphs: &[
"Legacy title, category, keywords, summary, description, behavior, examples, FAQs, links, source and test references are retained here or in typed declarations. Fusion flags and provider precisions are represented by the canonical placement contract; no feature is required. Display-only examples are explanatory; executable examples below carry value assertions."
] }],
 examples: &[BuiltinExample { id: "array-write", title: "array-write",
 program: "S = struct(); S.type = '()'; S.subs = {uint64(2)}; value = subsasgn(uint64([4 9 2]), S, uint64(8));", display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(value,uint64([4 8 2])));" } }],
 example_exemption: None,
 faqs: &[BuiltinDocumentationFaq { question: "Should I call `subsasgn` directly in normal code?", answer: "Usually no. Prefer ordinary assignment syntax like `obj.field = value` or `obj(2) = value`. Direct calls are mainly for advanced dispatch and testing." },
BuiltinDocumentationFaq { question: "How is the assignment path represented?", answer: "Use `substruct` to build `S`. Each ordered element contains a `type` and `subs` field, and chained assignment uses multiple elements." },
BuiltinDocumentationFaq { question: "What does the global `subsasgn` actually do?", answer: "It resolves a class overload when present and otherwise executes the ordered path using canonical default indexing." }],
 links: &[BuiltinDocumentationLink { label: "subsref", target: BuiltinDocumentationLinkTarget::Documentation("./subsref") },
BuiltinDocumentationLink { label: "datetime", target: BuiltinDocumentationLinkTarget::Documentation("./datetime") },
BuiltinDocumentationLink { label: "duration", target: BuiltinDocumentationLinkTarget::Documentation("./duration") },
BuiltinDocumentationLink { label: "containers.Map", target: BuiltinDocumentationLinkTarget::Documentation("./containers.map") }],
 media: &[],
 evidence: BuiltinDocumentationEvidence {
 implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/object_indexing.rs") }],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Semantic regression coverage", location: "runmat-ignition/tests/functions.rs" }], notes: &[],
 }, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
