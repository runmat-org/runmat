use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {

    authority: BuiltinDocumentationAuthority::Catalog,

    title: Some("underlyingType"),

    slug: Some("underlyingType"),

    summary: "Return the underlying MATLAB data type that determines array behavior.",

    description: "`underlyingType(X)` returns the name of the underlying data type that determines how `X` behaves. For most values this matches `class(X)`. For `gpuArray` values, `class(X)` returns `\"gpuArray\"` while `underlyingType(X)` returns the resident element type such as `\"double\"`, `\"single\"`, `\"int32\"`, `\"uint8\"`, or `\"logical\"`.",

    keywords: &[
"underlyingType",
 "underlying data type",
 "type inspection",
 "gpuArray",
 "class"]
,

    related: &[
]
,

    sections: &[

        BuiltinDocumentationSection {
 heading: "Behavior",
 paragraphs: &[
"Double scalars, complex values, sparse arrays, and double tensors report `\"double\"`.",
 "Single-precision tensors report `\"single\"`; all eight integer classes report their exact signedness and width, such as `\"int32\"` or `\"uint16\"`.",
 "Logical scalars and logical arrays report `\"logical\"`. Character arrays report `\"char\"`, and string scalars or arrays report `\"string\"`.",
 "Cell arrays, structs, objects, handles, listeners, function handles, and metadata values report the same name as `class` because their behavior is not modeled by a different underlying numeric element type.",
 "`gpuArray` inputs are metadata-only: RunMat validates coherent floating, logical, or integer class metadata and does not copy the device buffer back to the host."]
 }
,

        BuiltinDocumentationSection {
 heading: "GPU and residency",
 paragraphs: &[
"For gpuArray values uploaded through RunMat, exact class-name metadata is preferred so integer device arrays report names such as `\"int32\"` or `\"uint8\"`. If only provider precision is available, F32 maps to `\"single\"` and F64 or unannotated numeric handles map to `\"double\"`.",
 "`underlyingType` is metadata-only for gpuArray inputs. It does not launch kernels, copy buffers, or change residency.",
 "No GPU kernels run. RunMat inspects gpuArray class-name, logical, and precision metadata without gathering data."]
 }
,

        BuiltinDocumentationSection {
 heading: "Legacy examples and displayed results",
 paragraphs: &[
"Query a scalar double\nt = underlyingType(42)\nExpected display: t = \"double\"",
 "Query a single-precision array\nA = single([1 2 3]);\nt = underlyingType(A)\nExpected display: t = \"single\"",
 "Compare class and underlying type for gpuArray data\nG = gpuArray(single([1 2 3]));\nc = class(G);\nt = underlyingType(G)\nExpected display: c = \"gpuArray\"\nt = \"single\""]
 }
,

    ]
,

    examples: &[
BuiltinExample {
 id: "typed-basic",
 title: "Inspect a type with an assertion",
 program: "t = underlyingType(42)",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::Assertions {
 source: "assert(strcmp(t,\"double\"));" }
 }
]
,

    example_exemption: None,

    faqs: &[
BuiltinDocumentationFaq {
 question: "How is `underlyingType` different from `class`?",
 answer: "For ordinary RunMat values they usually match. For gpuArray values, `class` reports `\"gpuArray\"` while `underlyingType` reports the underlying element type that controls numeric behavior." }
,
 BuiltinDocumentationFaq {
 question: "Does `underlyingType` gather gpuArray data?",
 answer: "No. RunMat reads handle metadata and leaves device buffers resident." }
,
 BuiltinDocumentationFaq {
 question: "What does it return for objects and containers?",
 answer: "Objects and containers such as cells and structs return the same name as `class` unless RunMat has a distinct underlying storage type for that value family." }
]
,

    links: &[
BuiltinDocumentationLink {
 label: "class",
 target: BuiltinDocumentationLinkTarget::Documentation("./class") }
,
 BuiltinDocumentationLink {
 label: "isUnderlyingType",
 target: BuiltinDocumentationLinkTarget::Documentation("./isUnderlyingType") }
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
