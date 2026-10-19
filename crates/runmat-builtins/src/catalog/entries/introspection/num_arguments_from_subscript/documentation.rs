use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog,
 title: Some("numArgumentsFromSubscript"), slug: Some("numArgumentsFromSubscript"),
 summary: "Return the expected argument count for MATLAB-style indexing described by a substruct.",
 description: "`numArgumentsFromSubscript(A, S, indexingContext)` returns the number of expected outputs from `subsref` or expected inputs to `subsasgn` for an indexing expression. `S` is a substruct-compatible descriptor with `type` and `subs` fields, and `indexingContext` is `Statement`, `Expression`, or `Assignment` through `matlab.indexing.IndexingContext` or the legacy `matlab.mixin.util.IndexingContext` name.",
 keywords: &["numArgumentsFromSubscript",
"substruct",
"subsref",
"subsasgn",
"object indexing",
"IndexingContext"], related: &[],
 sections: &[BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
"The public signature is `n = numArgumentsFromSubscript(A, S, indexingContext)`.",
"`S.type` accepts `\"()\"`, `\".\"`, or `\"{}\"` as a string scalar or row character vector.",
"`S.subs` is text for dot indexing and a cell array of index vectors for parentheses or brace indexing.",
"Scalar structures describe one indexing level. A structure array describes multiple levels; every element has `type` and `subs` fields.",
"Object and handle-object targets dispatch to the receiver class's `numArgumentsFromSubscript` method when it is registered. If no overload exists, RunMat falls back to default array behavior.",
"For ordinary arrays, parentheses and dot indexing report one argument. Brace indexing reports the number of selected elements, including validated numeric vector selectors, logical selectors, `':'`, and multidimensional selector products.",
"All eight integer target classes and integer scalar/vector subscripts use authoritative shape or typed element storage. Wide positive `uint64` selectors are validated without binary64 conversion, and the structural result crosses to the documented double scalar only when exactly representable.",
"`matlab.indexing.IndexingContext.Statement`, `.Expression`, and `.Assignment` are registered lazily when used through `classref`, and the legacy `matlab.mixin.util.IndexingContext` class name resolves to equivalent enum members.",
"RunMat also accepts compatible text tokens such as `'Statement'` and `'matlab.indexing.IndexingContext.Expression'` for integration with code that constructs context values dynamically.",
"Invalid substructs and unsupported context values raise stable `RunMat:numArgumentsFromSubscript:*` identifiers."
] },
BuiltinDocumentationSection { heading: "Execution and ownership", paragraphs: &[
"Default indexing uses the shared object protocol. Class methods may throw, suspend, mutate handles, access ambient services or invoke arbitrary code. Unknown overload effects and capabilities remain conservative.",
"Native and interpreted execution share the portable runtime protocol and process-stack reentry. Object/class registry dependencies and callback captures follow shared GC rooting and affinity rules; retries cannot replay overload effects.",
"Direct distributed inputs are unsupported. Ordinary workers require transferable values, available class metadata and callable/service dependencies; resident and foreign members retain their owner and affinity requirements.",
"Shared runtime placement owns timing and transfer accounting; this identity adds no scheduler or hardware thresholds."
] },
BuiltinDocumentationSection { heading: "GPU behavior", paragraphs: &[
"Metadata-only indexing introspection. GPU tensor targets use handle shape metadata for default brace counts and are not gathered.",
"For gpuArray targets, RunMat uses `GpuTensorHandle.shape` to count colon-selected brace outputs.",
"For GPU-resident numeric index vectors, RunMat returns `RunMat:numArgumentsFromSubscript:InvalidSubstruct` rather than gathering selector values.",
"The default `numArgumentsFromSubscript` path is host-side metadata logic without target gathers or provider allocations. Class overloads may execute arbitrary code and return values with different classes or residency."
] },
BuiltinDocumentationSection { heading: "Additional usage", paragraphs: &[
"Count outputs from brace indexing\nC = {\"one\", 2, \"three\"};\nS = struct();\nS.type = '{}';\nS.subs = {[1 2]};\nn = numArgumentsFromSubscript(C, S, 'Statement')\nExpected display: n = 2",
"Use the IndexingContext enum\nC = {\"one\", 2, \"three\"};\nS = struct();\nS.type = '{}';\nS.subs = {':'};\nctx = classref('matlab.indexing.IndexingContext').Statement;\nn = numArgumentsFromSubscript(C, S, ctx)\nExpected display: n = 3"
] },
BuiltinDocumentationSection { heading: "Documentation reconciliation", paragraphs: &[
"Legacy title, category, keywords, summary, description, behavior, examples, FAQs, links, source and test references are retained here or in typed declarations. Fusion flags and provider precisions are represented by the canonical placement contract; no feature is required. Display-only examples are explanatory; executable examples below carry value assertions."
] }],
 examples: &[BuiltinExample { id: "brace-count", title: "brace-count",
 program: "C = {1,2,3}; S = struct(); S.type = '{}'; S.subs = {[1 3]}; n = numArgumentsFromSubscript(C,S,'Statement');", display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(n,2));" } },
BuiltinExample { id: "enum-count", title: "enum-count",
 program: "C = {1,2,3}; S = struct(); S.type = '{}'; S.subs = {':'}; ctx = classref('matlab.indexing.IndexingContext').Statement; n = numArgumentsFromSubscript(C,S,ctx);", display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(n,3));" } }],
 example_exemption: None,
 faqs: &[BuiltinDocumentationFaq { question: "Does RunMat implement the `substruct` helper itself?", answer: "Yes. The public `substruct` helper constructs standard paths; structures built explicitly with type and subs fields are also accepted." },
BuiltinDocumentationFaq { question: "Can user classes override the result?", answer: "Yes. For object and handle-object targets, RunMat dispatches to a registered `Class.numArgumentsFromSubscript` method before applying the default array behavior." },
BuiltinDocumentationFaq { question: "Does this gather gpuArray data?", answer: "No. The default path only reads target shape metadata from `GpuTensorHandle` values when counting `':'`. GPU-resident index vectors are rejected because their values cannot be validated without gathering." }],
 links: &[BuiltinDocumentationLink { label: "subsref", target: BuiltinDocumentationLinkTarget::Documentation("./subsref") },
BuiltinDocumentationLink { label: "subsasgn", target: BuiltinDocumentationLinkTarget::Documentation("./subsasgn") },
BuiltinDocumentationLink { label: "classref", target: BuiltinDocumentationLinkTarget::Documentation("./classref") },
BuiltinDocumentationLink { label: "struct", target: BuiltinDocumentationLinkTarget::Documentation("./struct") }],
 media: &[],
 evidence: BuiltinDocumentationEvidence {
 implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/object_indexing.rs") }],
 verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Semantic regression coverage", location: "builtins::introspection::object_indexing::tests" },
BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Semantic regression coverage", location: "crates/runmat-vm/tests/functions.rs::num_arguments_from_subscript_script_surface" }], notes: &[],
 }, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
