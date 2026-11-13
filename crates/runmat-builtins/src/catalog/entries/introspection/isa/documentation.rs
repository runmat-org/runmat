use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {

    authority: BuiltinDocumentationAuthority::Catalog,

    title: Some("isa"),

    slug: Some("isa"),

    summary: "Test class/category membership in MATLAB and RunMat.",

    description: "`isa(x, T)` returns true when value `x` belongs to class or abstract category `T`. Supported class/category checks and type coercions follow MATLAB semantics.",

    keywords: &[
"isa",
 "type checking",
 "class comparison",
 "numeric category",
 "gpuArray"]
,

    related: &[
]
,

    sections: &[

        BuiltinDocumentationSection {
 heading: "Behavior",
 paragraphs: &[
"Exact class names (such as `\"double\"`, `\"cell\"`, `\"struct\"`, `\"event.listener\"`) match the value’s dynamic class. Comparisons are case-insensitive.",
 "Abstract categories (`\"numeric\"`, `\"float\"`, `\"integer\"`, `\"handle\"`, `\"function_handle\"`, `\"gpuArray\"`, `\"listener\"`) mirror MATLAB’s grouping rules.",
 "Logical scalars and logical arrays satisfy `\"logical\"` and do not register as `\"numeric\"`. Internally auto-resident logical masks retain those host categories, while explicit gpuArray values expose the gpuArray wrapper instead.",
 "Handle inheritance is respected: if a class derives from `handle`, `isa(obj, \"handle\")` returns `true` even when `class(obj)` reports a subclass name.",
 "`isa` accepts either a string scalar or a character row vector as the class designator.",
 "Resident inputs never need to be gathered. Only explicit user-visible residency matches `\"gpuArray\"`; internal automatic residency matches its underlying host class and categories."]
 }
,

        BuiltinDocumentationSection {
 heading: "GPU and residency",
 paragraphs: &[
"`isa` never launches GPU kernels. Explicit gpuArray inputs satisfy only the public wrapper identity `\"gpuArray\"`; internal automatic residency remains transparent and satisfies its underlying host class and categories. Because the builtin returns a host logical scalar from metadata, it does not participate in fusion and providers need no additional hooks.",
 concat!(
"You usually do NOT need to call `gpuArray` yourself in RunMat (unlike MATLAB).\n\nIn RunMat, the fusion planner keeps residency on GPU in branches of fused expressions. As such, when you call `isa` on a gpuArray result, the planner preserves the device buffer and `isa` answers the query using metadata only.\n\nTo preserve backwards compatibility with MathWorks MATLAB, and for when you want to explicitly bootstrap GPU residency, you can call `gpuArray` explicitly to move data to the GPU if you want to be explicit about the residency.\n\nSince MathWorks MATLAB does not have a fusion planner, and they kept their parallel execution toolbox separate from the core language, as their toolbox is a ",
"separate commercial product, MathWorks MATLAB users need to call `gpuArray` to move data to the GPU manually whereas RunMat users can rely on the fusion planner to keep data on the GPU automatically.",
),
 "Returns host logical scalars and inspects resident metadata without launching kernels. Explicit gpuArray values expose wrapper identity; internal automatic residency exposes underlying host class/category semantics."]
 }
,

        BuiltinDocumentationSection {
 heading: "Legacy examples and displayed results",
 paragraphs: &[
"Checking whether a scalar is double precision\ntf = isa(42, \"double\")\nExpected display: tf = logical(1)",
 "Testing numeric arrays against the numeric category\nA = rand(3, 4);\nis_numeric = isa(A, \"numeric\");\nis_integer = isa(A, \"integer\")\nExpected display: is_numeric = logical(1)\nis_integer = logical(0)",
 "Checking explicit gpuArray wrapper identity\nG = gpuArray(rand(1024, 1024));\ntf_gpu = isa(G, \"gpuArray\");\ntf_numeric = isa(G, \"numeric\")\nExpected display: tf_gpu = logical(1)\ntf_numeric = logical(0)",
 "Inspecting the underlying type of an explicit GPU mask\nG = gpuArray(rand(5) > 0.5);\nis_wrapper = isa(G, \"gpuArray\");\nis_mask = isUnderlyingType(G, \"logical\")\nExpected display: is_wrapper = logical(1)\nis_mask = logical(1)",
 "Validating handle subclasses\npt = pkg.TestHandle();          % derives from handle\ntf_handle = isa(pt, \"handle\");\ntf_exact = isa(pt, \"pkg.TestHandle\")\nExpected display: tf_handle = logical(1)\ntf_exact = logical(1)",
 "Comparing string scalars and character vectors\ns = \"RunMat\";\nc = 'RunMat';\ntf_string = isa(s, \"string\");\ntf_char = isa(c, \"char\")\nExpected display: tf_string = logical(1)\ntf_char = logical(1)",
 "Detecting function handles and listeners\nfh = @sin;\ntf_handle = isa(fh, \"function_handle\");\nlst = addlistener(TestSource, \"Changed\", @disp);\ntf_listener = isa(lst, \"listener\")\nExpected display: tf_handle = logical(1)\ntf_listener = logical(1)"]
 }
,

    ]
,

    examples: &[
BuiltinExample {
 id: "typed-basic",
 title: "Inspect a type with an assertion",
 program: "tf = isa(42,\"double\")",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(tf);" }
 }
]
,

    example_exemption: None,

    faqs: &[
BuiltinDocumentationFaq {
 question: "Does `isa` treat gpuArray values as numeric?",
 answer: "An explicit gpuArray exposes its wrapper identity and matches `\"gpuArray\"`, not the underlying numeric categories. RunMat's internal automatic residency remains transparent and matches the underlying host class and categories. Use `isUnderlyingType` to inspect an explicit gpuArray's element type." }
,
 BuiltinDocumentationFaq {
 question: "How do I check for integer types?",
 answer: "Use the `\"integer\"` category. It matches MATLAB’s native integer classes (`int8`, `uint16`, and so on). For example `isa(int32(5), \"integer\")` returns `true` while `isa(5, \"integer\")` returns `false`." }
,
 BuiltinDocumentationFaq {
 question: "Does `isa` understand handle inheritance?",
 answer: "Yes. If a class derives from `handle`, then the object passes the `\"handle\"` category check. The builtin walks parent classes registered with RunMat to mirror MATLAB semantics." }
,
 BuiltinDocumentationFaq {
 question: "What kinds of strings can I pass as the type name?",
 answer: "Provide either a string scalar (double quotes) or a character row vector (single quotes). Multi-row character arrays and string arrays with more than one element throw a descriptive error." }
,
 BuiltinDocumentationFaq {
 question: "How do I detect function handles or anonymous functions?",
 answer: "Both named function handles and closures pass `isa(value, \"function_handle\")`." }
,
 BuiltinDocumentationFaq {
 question: "Will `isa` gather gpuArray inputs or launch kernels?",
 answer: "No. The builtin is metadata-only and returns a host logical scalar without copying device buffers." }
,
 BuiltinDocumentationFaq {
 question: "How are listeners treated?",
 answer: "Event listeners pass both `isa(listener, \"event.listener\")` and the compatibility alias `isa(listener, \"listener\")`. They also satisfy the `\"handle\"` category." }
,
 BuiltinDocumentationFaq {
 question: "Does `isa` differentiate between string and char arrays?",
 answer: "Yes. String scalars pass `\"string\"` while character arrays pass `\"char\"`. They are not interchangeable." }
,
 BuiltinDocumentationFaq {
 question: "Can I use `isa` to test meta-class values?",
 answer: "Yes. Meta-class references created with `classref` return `true` for `\"meta.class\"`." }
]
,

    links: &[
BuiltinDocumentationLink {
 label: "`class`",
 target: BuiltinDocumentationLinkTarget::Documentation("./class") }
,
 BuiltinDocumentationLink {
 label: "`isnumeric`",
 target: BuiltinDocumentationLinkTarget::Documentation("./isnumeric") }
,
 BuiltinDocumentationLink {
 label: "`islogical`",
 target: BuiltinDocumentationLinkTarget::Documentation("./islogical") }
,
 BuiltinDocumentationLink {
 label: "`gpuArray`",
 target: BuiltinDocumentationLinkTarget::Documentation("./gpuarray") }
,
 BuiltinDocumentationLink {
 label: "`gather`",
 target: BuiltinDocumentationLinkTarget::Documentation("./gather") }
,
 BuiltinDocumentationLink {
 label: "ischar",
 target: BuiltinDocumentationLinkTarget::Documentation("./ischar") }
,
 BuiltinDocumentationLink {
 label: "isstring",
 target: BuiltinDocumentationLinkTarget::Documentation("./isstring") }
,
 BuiltinDocumentationLink {
 label: "which",
 target: BuiltinDocumentationLinkTarget::Documentation("./which") }
,
 BuiltinDocumentationLink {
 label: "who",
 target: BuiltinDocumentationLinkTarget::Documentation("./who") }
,
 BuiltinDocumentationLink {
 label: "whos",
 target: BuiltinDocumentationLinkTarget::Documentation("./whos") }
]
,

    media: &[
]
,

    evidence: BuiltinDocumentationEvidence {
 implementation: &[
BuiltinDocumentationLink {
 label: "Runtime implementation",
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/isa.rs") }
]
,
 verification: &[
BuiltinEvidenceReference {
 kind: BuiltinEvidenceKind::UnitTest,
 label: "Runtime semantics",
 location: "builtins::introspection::isa::tests" }
]
,
 notes: &[
]
 }
,

    introduced: None,

    status: Some(BuiltinDocumentationStatus::Partial),

}
;
