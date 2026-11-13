use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {

    authority: BuiltinDocumentationAuthority::Catalog,

    title: Some("isobject"),

    slug: Some("isobject"),

    summary: "Return true when a value is a MATLAB object.",

    description: "`isobject(A)` returns true when `A` is represented as a MATLAB class object in RunMat, including value objects, handle objects, listeners, metaclass references, and `MException` values. Plotting handles are still represented by RunMat's numeric graphics-handle compatibility layer, so use `ishandle`/graphics APIs for those until graphics handles move to object values.",

    keywords: &[
"isobject",
 "object",
 "handle",
 "classdef",
 "type checking"]
,

    related: &[
]
,

    sections: &[

        BuiltinDocumentationSection {
 heading: "Behavior",
 paragraphs: &[
"Value-class and handle-class instances return `logical(1)`.",
 "Runtime object values such as event listeners, metaclass references, MException instances, and explicit gpuArray handles return `logical(1)` because they expose MATLAB class/object semantics.",
 "Numeric, logical, character, string, cell, struct, and function-handle values return `logical(0)`.",
 "Current RunMat plotting handles are numeric compatibility handles rather than `Value::HandleObject` instances, so `isobject` does not treat figure, axes, or plot-child handles as objects yet.",
 "The predicate does not inspect object properties or call user code."]
 }
,

        BuiltinDocumentationSection {
 heading: "GPU and residency",
 paragraphs: &[
"Metadata-only predicate. Explicit gpuArray values are MATLAB objects and remain resident; internal automatic residency retains the underlying host value's nonobject semantics."]
 }
,

        BuiltinDocumentationSection {
 heading: "Legacy examples and displayed results",
 paragraphs: &[
"Check ordinary numeric data\ntf = isobject(42)\nExpected display: tf = logical(0)",
 "Check a class object\nobj = classref(\"Point\").origin();\ntf = isobject(obj)\nExpected display: tf = logical(1)",
 "Function handles are not objects\ntf = isobject(@sin)\nExpected display: tf = logical(0)"]
 }
,

    ]
,

    examples: &[
BuiltinExample {
 id: "typed-basic",
 title: "Inspect a type with an assertion",
 program: "tf = isobject(42)",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(~tf);" }
 }
]
,

    example_exemption: None,

    faqs: &[
BuiltinDocumentationFaq {
 question: "Does `isobject` call class methods?",
 answer: "No. It only inspects runtime type metadata and never invokes user-defined code." }
,
 BuiltinDocumentationFaq {
 question: "Are structs objects?",
 answer: "No. MATLAB structs are data containers, not class objects, so `isobject(struct(...))` returns false." }
,
 BuiltinDocumentationFaq {
 question: "What about gpuArray?",
 answer: "An explicit `gpuArray(...)` value follows MATLAB's object classification and returns true without gathering. RunMat's internal automatic residency is transparent and retains the underlying host value's nonobject result." }
,
 BuiltinDocumentationFaq {
 question: "What about graphics handles?",
 answer: "RunMat currently represents figure, axes, and plot-child handles as numeric compatibility handles. Use `ishandle`, `get`, `set`, and graphics query functions for those handles; object-valued graphics handles require a broader graphics-handle representation change." }
]
,

    links: &[
BuiltinDocumentationLink {
 label: "class",
 target: BuiltinDocumentationLinkTarget::Documentation("./class") }
,
 BuiltinDocumentationLink {
 label: "isa",
 target: BuiltinDocumentationLinkTarget::Documentation("./isa") }
,
 BuiltinDocumentationLink {
 label: "isstring",
 target: BuiltinDocumentationLinkTarget::Documentation("./isstring") }
,
 BuiltinDocumentationLink {
 label: "ishandle",
 target: BuiltinDocumentationLinkTarget::Documentation("./ishandle") }
]
,

    media: &[
]
,

    evidence: BuiltinDocumentationEvidence {
 implementation: &[
BuiltinDocumentationLink {
 label: "Runtime implementation",
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/isobject.rs") }
]
,
 verification: &[
BuiltinEvidenceReference {
 kind: BuiltinEvidenceKind::UnitTest,
 label: "Runtime semantics",
 location: "builtins::introspection::isobject::tests" }
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
