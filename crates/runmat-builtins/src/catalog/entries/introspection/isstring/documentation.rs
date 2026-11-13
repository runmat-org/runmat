use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {

    authority: BuiltinDocumentationAuthority::Catalog,

    title: Some("isstring"),

    slug: Some("isstring"),

    summary: "Test whether values are string arrays in MATLAB and RunMat.",

    description: "`tf = isstring(x)` returns true when `x` is a string array (including string scalars) and false otherwise. Distinction from char and other types follows MATLAB semantics.",

    keywords: &[
"isstring",
 "string array",
 "type checking",
 "string scalar",
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
"String scalars such as `\"RunMat\"` and string arrays like `[\"alpha\", \"beta\"]` both return `logical(1)`.",
 "Empty string arrays produced by `strings(0)` or `string.empty(...)` are still recognised as the string type, so they return `logical(1)`.",
 "Character arrays created with single quotes (`'text'`), cell arrays of character vectors, numeric tensors, logical masks, structs, tables, and other non-string containers return `logical(0)`.",
 "gpuArray inputs currently store numeric or logical tensors in RunMat, so `isstring(gpuArray(...))` returns `logical(0)`, matching MATLAB’s behaviour.",
 "The result is always a host logical scalar, making it easy to use for argument validation, control-flow guards, or input parsing."]
 }
,

        BuiltinDocumentationSection {
 heading: "GPU and residency",
 paragraphs: &[
"`isstring` runs entirely from metadata. When you pass a gpuArray, RunMat inspects the handle and returns `logical(0)` because device buffers do not yet store string data. No kernels are launched, and no host↔device transfers occur, so the device allocation remains resident for subsequent operations.",
 concat!(
"You usually do NOT need to call `gpuArray` yourself in RunMat (unlike MATLAB).\n\nIn RunMat, the fusion planner keeps residency on GPU in branches of fused expressions. As such, calling `isstring` on a gpuArray result leaves the device allocation untouched while still answering the question using metadata only.\n\nTo preserve backwards compatibility with MathWorks MATLAB, and for when you want to explicitly bootstrap GPU residency, you can call `gpuArray` explicitly to move data to the GPU if you want to be explicit about the residency.\n\nSince MathWorks MATLAB does not have a fusion planner, and they kept their parallel execution toolbox separate from the core language, as their toolbox is ",
"a separate commercial product, MathWorks MATLAB users need to call `gpuArray` to move data to the GPU manually whereas RunMat users can rely on the fusion planner to keep data on the GPU automatically.",
),
 "Inspects metadata on the host; gpuArray inputs are classified without kernel launches."]
 }
,

        BuiltinDocumentationSection {
 heading: "Legacy examples and displayed results",
 paragraphs: &[
"Checking if a string scalar is recognised\ntf = isstring(\"RunMat\")\nExpected display: tf = logical(1)",
 "Detecting string arrays with multiple elements\nvalues = [\"alpha\", \"beta\", \"gamma\"];\ntf = isstring(values)\nExpected display: tf = logical(1)",
 "Distinguishing strings from character vectors\ntf_string = isstring(\"hello\");\ntf_char   = isstring('hello')\nExpected display: tf_string = logical(1)\ntf_char   = logical(0)",
 "Confirming that empty string arrays are still strings\nemptyStrs = strings(0, 2);\ntf = isstring(emptyStrs)\nExpected display: tf = logical(1)",
 "Recognising that numeric and logical arrays are not strings\nnumbers = [1 2 3];\nmask = true(1, 3);\ntf_numbers = isstring(numbers);\ntf_mask = isstring(mask)\nExpected display: tf_numbers = logical(0)\ntf_mask = logical(0)",
 "Validating cell arrays of character vectors\ncells = {'a', 'b', 'c'};\ntf = isstring(cells)\nExpected display: tf = logical(0)",
 "Inspecting gpuArray inputs without transfers\nG = gpuArray(ones(2, 2));\ntf = isstring(G)\nExpected display: tf = logical(0)",
 "Using `isstring` in argument validation\nfunction greet(name)\n    if ~isstring(name)\n        error(\"Name must be provided as a string.\");\n    end\n    disp(\"Hello \" + name + \"!\");\nend\n\ngreet(\"RunMat\");\n% greet(42); % throws \"Name must be provided as a string.\"\nExpected display: Hello RunMat!"]
 }
,

    ]
,

    examples: &[
BuiltinExample {
 id: "typed-basic",
 title: "Inspect a type with an assertion",
 program: "tf = isstring(\"RunMat\")",
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
 question: "Does `isstring` return true for empty string arrays?",
 answer: "Yes. Empty string arrays are still of type string, so `isstring(strings(0))` returns `logical(1)`." }
,
 BuiltinDocumentationFaq {
 question: "How is `isstring` different from `ischar`?",
 answer: "`isstring` recognises MATLAB’s modern string type introduced in R2016b. `ischar` recognises legacy character arrays created with single quotes. Only `isstring` returns `true` for double-quoted values." }
,
 BuiltinDocumentationFaq {
 question: "What about string arrays stored inside cell arrays?",
 answer: "Cell arrays are containers, so `isstring({'a', 'b'})` returns `logical(0)`. Use `iscellstr` or `all(isstring(cells))` to validate the contents instead." }
,
 BuiltinDocumentationFaq {
 question: "Can gpuArray hold string data in RunMat?",
 answer: "Not currently. gpuArray values hold numeric or logical tensors, so `isstring` always reports `logical(0)` for device-resident data." }
,
 BuiltinDocumentationFaq {
 question: "Does `isstring` inspect array contents element by element?",
 answer: "No. It only checks the value’s type metadata, so the cost is constant regardless of array size." }
]
,

    links: &[
BuiltinDocumentationLink {
 label: "ischar",
 target: BuiltinDocumentationLinkTarget::Documentation("./ischar") }
,
 BuiltinDocumentationLink {
 label: "isa",
 target: BuiltinDocumentationLinkTarget::Documentation("./isa") }
,
 BuiltinDocumentationLink {
 label: "string",
 target: BuiltinDocumentationLinkTarget::Documentation("./string") }
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
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/isstring.rs") }
]
,
 verification: &[
BuiltinEvidenceReference {
 kind: BuiltinEvidenceKind::UnitTest,
 label: "Runtime semantics",
 location: "builtins::introspection::isstring::tests" }
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
