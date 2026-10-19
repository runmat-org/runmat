use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {

    authority: BuiltinDocumentationAuthority::Catalog,

    title: Some("ischar"),

    slug: Some("ischar"),

    summary: "Test whether a value is a character array in MATLAB and RunMat.",

    description: "`tf = ischar(x)` returns true when `x` is a character array (char row vector or char matrix) and false otherwise. Distinction from string arrays and other types follows MATLAB semantics.",

    keywords: &[
"ischar",
 "character arrays",
 "type checking",
 "char",
 "logical predicate",
 "gpuArray"]
,

    related: &[
]
,

    sections: &[

        BuiltinDocumentationSection {
 heading: "Behavior",
 paragraphs: &[
"Character vectors such as `'RunMat'` and char matrices created with `['A'; 'B']` return `true`.",
 "Empty char arrays (including `''` and `char.empty(...)`) still count as character arrays.",
 "String scalars, string arrays, cell arrays, numeric values, and logical masks all return `false`.",
 "gpuArray inputs return `false` unless they are explicitly stored as char arrays (which RunMat does not currently support), matching MATLAB’s behaviour where gpuArray does not host char data.",
 "The result is always a host logical scalar (`logical(0)` or `logical(1)`), so it can be used in branching, masking, or validation code without additional conversions."]
 }
,

        BuiltinDocumentationSection {
 heading: "GPU and residency",
 paragraphs: &[
"`ischar` never launches GPU kernels. When you pass a gpuArray value, RunMat inspects the residency metadata and simply reports `false` because device-resident buffers are numeric or logical. This matches MATLAB’s semantics and avoids unnecessary host↔device transfers.",
 concat!(
"You usually do NOT need to call `gpuArray` yourself in RunMat (unlike MATLAB).\n\nIn RunMat, the fusion planner keeps residency on GPU in branches of fused expressions. As such, calling `ischar` on a gpuArray result preserves the device buffer, and `ischar` answers the query using metadata only.\n\nTo preserve backwards compatibility with MathWorks MATLAB, and for when you want to explicitly bootstrap GPU residency, you can call `gpuArray` explicitly to move data to the GPU if you want to be explicit about the residency.\n\nSince MathWorks MATLAB does not have a fusion planner, and they kept their parallel execution toolbox separate from the core language, as their toolbox is a separate ",
"commercial product, MathWorks MATLAB users need to call `gpuArray` to move data to the GPU manually whereas RunMat users can rely on the fusion planner to keep data on the GPU automatically.",
),
 "Returns a host logical scalar; GPU inputs are inspected without launching kernels."]
 }
,

        BuiltinDocumentationSection {
 heading: "Legacy examples and displayed results",
 paragraphs: &[
"Checking if a character vector is a char array\ntf = ischar('RunMat')\nExpected display: tf = logical(1)",
 "Detecting multi-row char matrices\nletters = ['ab'; 'cd'];\ntf = ischar(letters)\nExpected display: tf = logical(1)",
 "Distinguishing between char arrays and string scalars\ntf_char = ischar('hello');\ntf_string = ischar(\"hello\")\nExpected display: tf_char = logical(1)\ntf_string = logical(0)",
 "Recognising that numeric and logical data are not char arrays\nnumbers = [1 2 3];\nmask = true(1, 3);\ntf_numbers = ischar(numbers);\ntf_mask = ischar(mask)\nExpected display: tf_numbers = logical(0)\ntf_mask = logical(0)",
 "Testing gpuArray inputs\nG = gpuArray(ones(2, 2));\ntf_gpu = ischar(G)\nExpected display: tf_gpu = logical(0)"]
 }
,

    ]
,

    examples: &[
BuiltinExample {
 id: "typed-basic",
 title: "Inspect a type with an assertion",
 program: "tf = ischar('RunMat')",
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
 question: "Does `ischar` treat empty character arrays as char?",
 answer: "Yes. Both `''` and arrays created with `char.empty` return `logical(1)` because they are still character arrays, even when they have zero elements." }
,
 BuiltinDocumentationFaq {
 question: "Why does `ischar` return `false` for string scalars?",
 answer: "Strings in MATLAB are a distinct type introduced in R2016b. They are not interchangeable with char arrays, so `ischar(\"text\")` correctly returns `logical(0)`." }
,
 BuiltinDocumentationFaq {
 question: "What about cell arrays of characters?",
 answer: "Cell arrays such as `{'a', 'b'}` return `logical(0)` because they are cell containers rather than a single char array. Use `iscellstr` if you need to validate cell-of-char collections." }
,
 BuiltinDocumentationFaq {
 question: "Can gpuArray hold char data in RunMat?",
 answer: "Not currently. gpuArray values in RunMat contain numeric or logical tensors, so `ischar` reports `false` for all gpuArray inputs, mirroring MATLAB’s behaviour." }
,
 BuiltinDocumentationFaq {
 question: "How do I convert a string to a char array if `ischar` returns false?",
 answer: "Use `char(stringScalar)` or the `convertStringsToChars` utility when you need to operate on char arrays for legacy APIs." }
,
 BuiltinDocumentationFaq {
 question: "Is there any performance cost when checking large arrays?",
 answer: "No. `ischar` only inspects the value’s metadata; it does not copy or iterate over the array elements, so the cost is constant regardless of array size." }
]
,

    links: &[
BuiltinDocumentationLink {
 label: "isa",
 target: BuiltinDocumentationLinkTarget::Documentation("./isa") }
,
 BuiltinDocumentationLink {
 label: "char",
 target: BuiltinDocumentationLinkTarget::Documentation("./char") }
,
 BuiltinDocumentationLink {
 label: "string",
 target: BuiltinDocumentationLinkTarget::Documentation("./string") }
,
 BuiltinDocumentationLink {
 label: "strcmp",
 target: BuiltinDocumentationLinkTarget::Documentation("./strcmp") }
,
 BuiltinDocumentationLink {
 label: "gpuArray",
 target: BuiltinDocumentationLinkTarget::Documentation("./gpuarray") }
,
 BuiltinDocumentationLink {
 label: "class",
 target: BuiltinDocumentationLinkTarget::Documentation("./class") }
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
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/ischar.rs") }
]
,
 verification: &[
BuiltinEvidenceReference {
 kind: BuiltinEvidenceKind::UnitTest,
 label: "Runtime semantics",
 location: "builtins::introspection::ischar::tests" }
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
