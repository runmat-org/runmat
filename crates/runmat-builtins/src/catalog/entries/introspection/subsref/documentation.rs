use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("subsref"), slug: Some("subsref"),
 summary: "Apply a standard substruct indexing path to an indexable value.",
 description: "`subsref(A, S)` applies the nonempty structure-array path `S`, conventionally produced by `substruct`. Ordinary arrays, cells, strings, structures, and other indexable values use canonical default indexing; class objects may resolve a class-specific `subsref` implementation. Field, parenthesis, and brace steps remain ordered, and each overload receives the complete remaining path.",
 keywords: &["subsref",
"object indexing",
"overloaded indexing",
"dot indexing"], related: &[],
 sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
"The receiver may be any value that supports the requested default or overloaded indexing path.",
"Each element of `S` has a `type` field (`'.'`, `'()'`, or `'{}'`) and a corresponding `subs` field.",
"A multi-element `S` preserves the complete chained indexing path in source order.",
"Class objects with a `subsref` method receive the complete remaining path; values without an overload use canonical default indexing.",
"Integer selectors inside `S.subs` retain their native class, exact value, shape, and resident handle. Default indexing or the class method determines output class, conversion, and overflow behavior.",
"Prefer normal indexing syntax in user code; direct calls to `subsref` are mainly useful for advanced dispatch or testing."
] },
BuiltinDocumentationSection { heading: "Execution and ownership", paragraphs: &[
"Default indexing uses the shared object protocol. Class methods may throw, suspend, mutate handles, access ambient services or invoke arbitrary code. Unknown overload effects and capabilities remain conservative.",
"Native and interpreted execution share the portable runtime protocol and process-stack reentry. Object/class registry dependencies and callback captures follow shared GC rooting and affinity rules; retries cannot replay overload effects.",
"Direct distributed inputs are unsupported. Ordinary workers require transferable values, available class metadata and callable/service dependencies; resident and foreign members retain their owner and affinity requirements.",
"Shared runtime placement owns timing and transfer accounting; this identity adds no scheduler or hardware thresholds."
] },
BuiltinDocumentationSection { heading: "GPU behavior", paragraphs: &[
"Method resolution executes on the host, but selector payloads are forwarded to the class implementation without numeric conversion or an implicit residency change.",
"The shared default-indexing service owns numeric provider operations and materialization. Class overloads determine their own provider work; dispatch does not gather nested values.",
"`subsref` is host-side object dispatch and does not itself manage GPU residency."
] },
BuiltinDocumentationSection { heading: "Additional usage", paragraphs: &[
"Index a datetime array explicitly\nt = datetime([2024 2025], [1 6], [15 20]);\nS = substruct('()', {2});\nitem = subsref(t, S)\nExpected display: item =\n  20-Jun-2025 00:00:00",
"Read a datetime property explicitly\nt = datetime(2024, 4, 9);\nS = substruct('.', 'Format');\nfmt = subsref(t, S)\nExpected display: fmt =\n    'dd-MMM-yyyy HH:mm:ss'"
] },
BuiltinDocumentationSection { heading: "Documentation reconciliation", paragraphs: &[
"Legacy title, category, keywords, summary, description, behavior, examples, FAQs, links, source and test references are retained here or in typed declarations. Fusion flags and provider precisions are represented by the canonical placement contract; no feature is required. Display-only examples are explanatory; executable examples below carry value assertions."
] }],
 examples: &[BuiltinExample { id: "array-read", title: "array-read",
 program: "S = struct(); S.type = '()'; S.subs = {uint64(2)}; value = subsref(uint64([4 9 2]), S);", display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions { source: "assert(isa(value,'uint64')); assert(value == uint64(9));" } },
BuiltinExample { id: "cell-outputs", title: "cell-outputs",
 program: "S = struct(); S.type = '{}'; S.subs = {[1 2]}; [a,b] = subsref({uint64(7),'text'}, S);", display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions { source: "assert(isa(a,'uint64')); assert(a == uint64(7)); assert(strcmp(b,'text'));" } }],
 example_exemption: None,
 faqs: &[BuiltinDocumentationFaq { question: "Should I call `subsref` directly in normal code?", answer: "Usually no. Prefer ordinary syntax like `obj.field` or `obj(2)`. Direct calls are mainly for advanced dispatch and testing." },
BuiltinDocumentationFaq { question: "How is the indexing path represented?", answer: "Use `substruct` to build `S`. Each ordered element contains a `type` and `subs` field, and chained indexing uses multiple elements." },
BuiltinDocumentationFaq { question: "What does the global `subsref` actually do?", answer: "It resolves a class overload when present and otherwise executes the ordered path using canonical default indexing." }],
 links: &[BuiltinDocumentationLink { label: "subsasgn", target: BuiltinDocumentationLinkTarget::Documentation("./subsasgn") },
BuiltinDocumentationLink { label: "datetime", target: BuiltinDocumentationLinkTarget::Documentation("./datetime") },
BuiltinDocumentationLink { label: "duration", target: BuiltinDocumentationLinkTarget::Documentation("./duration") },
BuiltinDocumentationLink { label: "containers.Map", target: BuiltinDocumentationLinkTarget::Documentation("./containers.map") }],
 media: &[],
 evidence: BuiltinDocumentationEvidence {
 implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/object_indexing.rs") }],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Semantic regression coverage", location: "runmat-ignition/tests/functions.rs" }], notes: &[],
 }, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
