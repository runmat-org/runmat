use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {

    authority: BuiltinDocumentationAuthority::Catalog,

    title: Some("class"),

    slug: Some("class"),

    summary: "Return class names for values in MATLAB and RunMat.",

    description: "`class(x)` returns the class name for value `x` as a string scalar. Use it for diagnostics, branching, and display logic with MATLAB- and RunMat-compatible class labels.",

    keywords: &[
"class",
 "type inspection",
 "gpuArray class",
 "MATLAB class name",
 "isa"]
,

    related: &[
]
,

    sections: &[

        BuiltinDocumentationSection {
 heading: "Behavior",
 paragraphs: &[
"Real or complex double-precision inputs (including empty numeric arrays) report `\"double\"`. Integer scalars or tensors report their narrow class such as `\"int16\"` or `\"uint8\"`.",
 "Logical scalars and arrays report `\"logical\"`. Character arrays return `\"char\"`, while string scalars or arrays return `\"string\"`.",
 "Cell arrays, structs, and user-defined `classdef` objects report `\"cell\"`, `\"struct\"`, or the class name that defined the object.",
 "Function handles and anonymous closures both report `\"function_handle\"`. Listeners return `\"event.listener\"`.",
 "Explicit user-visible `gpuArray` values report `\"gpuArray\"` without gathering data. Internal automatic residency remains transparent and reports the underlying host class from coherent metadata.",
 "Handle objects, including deleted handles, return the class name that introduced the handle. Metadata-only values such as `classref(\"Point\")` report `\"meta.class\"`."]
 }
,

        BuiltinDocumentationSection {
 heading: "GPU and residency",
 paragraphs: &[
"`class` is an introspection-only builtin. When the input resides on the GPU, RunMat reads the residency metadata, returns the class name immediately, and leaves the data in place. No kernels are launched and no buffers are copied back to the CPU, so providers do not need to expose any hooks.",
 "You do not need to call `gpuArray` purely to query a value’s class. The auto-offload planner keeps tensors on the GPU whenever profitable, and `class` simply returns the residency-aware class name without altering where the data lives. Explicit `gpuArray` and `gather` calls remain available for compatibility with MATLAB code that manages residency manually.",
 "No GPU kernels run. Explicit user-visible gpuArray values report the wrapper class; internal automatic residency reports the underlying host class. Both paths use metadata without gathering data."]
 }
,

        BuiltinDocumentationSection {
 heading: "Legacy examples and displayed results",
 paragraphs: &[
"Check the class of a numeric scalar\nc = class(42)\nExpected display: c = \"double\"",
 "Inspect the class of a string array\nnames = [\"Ada\", \"Grace\", \"Edsger\"];\nclass_name = class(names)\nExpected display: class_name = \"string\"",
 "Determine whether a cell array stays typed as a cell\ncells = {1, \"two\", 3};\ncell_class = class(cells)\nExpected display: cell_class = \"cell\"",
 "Detect the class of gpuArray data without gathering\nG = gpuArray(rand(4));\ng_class = class(G)\nExpected display: g_class = \"gpuArray\"",
 "Report the class name of a custom object\nP = classref(\"Point\").origin();\npoint_class = class(P)\nExpected display: point_class = \"Point\""]
 }
,

    ]
,

    examples: &[
BuiltinExample {
 id: "typed-basic",
 title: "Inspect a type with an assertion",
 program: "c = class(42)",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(strcmp(c,\"double\"));" }
 }
]
,

    example_exemption: None,

    faqs: &[
BuiltinDocumentationFaq {
 question: "Does `class` return a string or a character array?",
 answer: "RunMat currently returns a string scalar. This preserves the existing runtime carrier; character-row parity with MATLAB remains a separate compatibility issue. Convert the result with `char(class(...))` when a character row is needed." }
,
 BuiltinDocumentationFaq {
 question: "How do I check whether a value is numeric?",
 answer: "Use `isa(x, \"numeric\")` to handle floats, complex numbers, and integers. `class` reports the concrete class (`\"double\"`, `\"int32\"`, and so on)." }
,
 BuiltinDocumentationFaq {
 question: "Will `class` gather gpuArray inputs?",
 answer: "No. RunMat inspects metadata without copying the buffer. Explicit gpuArray values return `\"gpuArray\"`; internally auto-resident values return their underlying host class." }
,
 BuiltinDocumentationFaq {
 question: "What does `class` return for handle objects?",
 answer: "`class` returns the object’s defining class name (for example `\"MyHandleClass\"`). Use `isa(h, \"handle\")` to test for handle semantics." }
,
 BuiltinDocumentationFaq {
 question: "What happens if I pass a struct or cell array?",
 answer: "Structs return `\"struct\"` and cell arrays return `\"cell\"`, matching MATLAB." }
,
 BuiltinDocumentationFaq {
 question: "What is the class of function handles?",
 answer: "Both named function handles and anonymous closures report `\"function_handle\"`." }
,
 BuiltinDocumentationFaq {
 question: "Can I distinguish string and char arrays?",
 answer: "Yes. `class(\"hello\")` yields `\"string\"` while `class('hello')` yields `\"char\"`." }
,
 BuiltinDocumentationFaq {
 question: "What about metadata objects such as `classref`?",
 answer: "Calling `class(classref(\"Point\"))` returns `\"meta.class\"`. Use the returned meta-class object to inspect properties or superclasses." }
]
,

    links: &[
BuiltinDocumentationLink {
 label: "isa",
 target: BuiltinDocumentationLinkTarget::Documentation("./isa") }
,
 BuiltinDocumentationLink {
 label: "underlyingType",
 target: BuiltinDocumentationLinkTarget::Documentation("./underlyingType") }
,
 BuiltinDocumentationLink {
 label: "isUnderlyingType",
 target: BuiltinDocumentationLinkTarget::Documentation("./isUnderlyingType") }
,
 BuiltinDocumentationLink {
 label: "isnumeric",
 target: BuiltinDocumentationLinkTarget::Documentation("./isnumeric") }
,
 BuiltinDocumentationLink {
 label: "gpuArray",
 target: BuiltinDocumentationLinkTarget::Documentation("./gpuarray") }
,
 BuiltinDocumentationLink {
 label: "gather",
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
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/class.rs") }
]
,
 verification: &[
BuiltinEvidenceReference {
 kind: BuiltinEvidenceKind::UnitTest,
 label: "Runtime semantics",
 location: "builtins::introspection::class::tests" }
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
