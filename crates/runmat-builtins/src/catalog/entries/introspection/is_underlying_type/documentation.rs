use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {

    authority: BuiltinDocumentationAuthority::Catalog,

    title: Some("isUnderlyingType"),

    slug: Some("isUnderlyingType"),

    summary: "Test whether a value has a specified underlying MATLAB data type.",

    description: "`isUnderlyingType(X, typename)` returns true when `underlyingType(X)` equals `typename`. The type name must be a string scalar or a character row vector.",

    keywords: &[
"isUnderlyingType",
 "underlying data type",
 "type checking",
 "gpuArray",
 "underlyingType"]
,

    related: &[
]
,

    sections: &[

        BuiltinDocumentationSection {
 heading: "Behavior",
 paragraphs: &[
"The predicate compares against the same canonical type names returned by `underlyingType`, such as `\"double\"`, `\"single\"`, `\"logical\"`, `\"char\"`, `\"string\"`, `\"cell\"`, and object class names.",
 "The `typename` input accepts a string scalar or a character row vector. Multi-element string arrays, multi-row character arrays, and non-text values throw a structured argument error.",
 "gpuArray inputs are checked from handle metadata. Exact class-name metadata is preferred for uploaded integer and floating-point device arrays; provider precision is used as a fallback for handles that only expose F32/F64.",
 "`isUnderlyingType(X, typename)` compares the requested type name exactly after trimming surrounding whitespace."]
 }
,

        BuiltinDocumentationSection {
 heading: "GPU and residency",
 paragraphs: &[
"No provider hook is required. RunMat maps gpuArray class-name, logical, and precision metadata to the same names returned by `underlyingType` and compares that name exactly with the requested type.",
 "`isUnderlyingType` is metadata-only for gpuArray inputs. It returns a host logical scalar and preserves provider residency for the inspected buffer.",
 "Returns a host logical scalar and inspects gpuArray metadata without launching kernels or gathering data."]
 }
,

        BuiltinDocumentationSection {
 heading: "Legacy examples and displayed results",
 paragraphs: &[
"Check a single-precision array\nA = single([1 2 3]);\ntf = isUnderlyingType(A, \"single\")\nExpected display: tf = logical(1)",
 "Reject a mismatched underlying type\ntf = isUnderlyingType(true, \"double\")\nExpected display: tf = logical(0)",
 "Check gpuArray element type without gathering\nG = gpuArray(single([1 2 3]));\ntf = isUnderlyingType(G, \"single\")\nExpected display: tf = logical(1)"]
 }
,

    ]
,

    examples: &[
BuiltinExample {
 id: "typed-basic",
 title: "Inspect a type with an assertion",
 program: "tf = isUnderlyingType(42,\"double\")",
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
 question: "Is this the same as `isa(X, typename)`?",
 answer: "No. `isa` tests the class or category of the value itself. `isUnderlyingType` tests the data type returned by `underlyingType`, which differs for gpuArray values." }
,
 BuiltinDocumentationFaq {
 question: "What text forms are accepted for `typename`?",
 answer: "Use a string scalar such as `\"double\"` or a character row vector such as `'double'`. Non-scalar string arrays and multi-row character arrays are rejected." }
,
 BuiltinDocumentationFaq {
 question: "Will this gather gpuArray data?",
 answer: "No. The predicate reads class-name, logical, and precision metadata from the gpuArray handle and returns a host logical scalar." }
]
,

    links: &[
BuiltinDocumentationLink {
 label: "underlyingType",
 target: BuiltinDocumentationLinkTarget::Documentation("./underlyingType") }
,
 BuiltinDocumentationLink {
 label: "class",
 target: BuiltinDocumentationLinkTarget::Documentation("./class") }
,
 BuiltinDocumentationLink {
 label: "isa",
 target: BuiltinDocumentationLinkTarget::Documentation("./isa") }
,
 BuiltinDocumentationLink {
 label: "gpuArray",
 target: BuiltinDocumentationLinkTarget::Documentation("./gpuarray") }
,
 BuiltinDocumentationLink {
 label: "gather",
 target: BuiltinDocumentationLinkTarget::Documentation("./gather") }
]
,

    media: &[
]
,

    evidence: BuiltinDocumentationEvidence {
 implementation: &[
BuiltinDocumentationLink {
 label: "Runtime implementation",
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/underlying_type.rs") }
]
,
 verification: &[
BuiltinEvidenceReference {
 kind: BuiltinEvidenceKind::UnitTest,
 label: "Runtime semantics",
 location: "builtins::introspection::underlying_type::tests" }
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
