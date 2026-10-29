use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {

    authority: BuiltinDocumentationAuthority::Catalog,

    title: Some("isvalid"),
 slug: Some("isvalid"),

    summary: "Return whether a handle object or listener is still valid.",

    description: "`isvalid(value)` returns a host logical scalar for a handle object or event listener. Non-handle inputs raise `RunMat:isvalid:InvalidInput` before any resident payload access.",

    keywords: &[
"isvalid",
 "handle",
 "listener",
 "validity",
 "classdef"]
,
 related: &[
"class",
 "isobject"]
,

    sections: &[
BuiltinDocumentationSection {
 heading: "Behavior and ownership",
 paragraphs: &[

        "Handle validity uses the original GC-backed target and deletion state. Deleted handles return false. For handle and listener receivers this sealed query does not dispatch an object method or consult ambient class declarations. Value classes retain ordinary method dispatch. Listener validity requires both live and enabled state; the query does not invoke a callback.",

        "Numeric, text, container, foreign and resident numeric inputs are invalid-domain inputs. The error retains the `isvalid` owner; resident buffers are not gathered or modified.",

        "The output is a host logical scalar. The query does not launch provider kernels or enter fusion; native and WASM execution use the same runtime path."
    ]
 }
]
,

    examples: &[
BuiltinExample {
 id: "invalid-input",
 title: "Reject a non-handle value",
 program: "tf = isvalid(42)",
 display_output: None,
 compatibility: BuiltinExampleCompatibility::Matlab,
 harness: BuiltinExampleHarness::Portable,
 fixture: BuiltinExampleFixture::None,
 requirements: BuiltinExampleRequirements::NONE,
 verification: BuiltinExampleVerification::ExpectedError {
 identifier: "RunMat:isvalid:InvalidInput" }
 }
]
,

    example_exemption: None,

    faqs: &[
BuiltinDocumentationFaq {
 question: "Are numeric arrays valid inputs?",
 answer: "No. This query accepts handle objects and listeners only." }
,
 BuiltinDocumentationFaq {
 question: "Does isvalid gather GPU arrays?",
 answer: "No. Resident numeric arrays reject from metadata before provider access." }
]
,

    links: &[
BuiltinDocumentationLink {
 label: "class",
 target: BuiltinDocumentationLinkTarget::Documentation("./class") }
,
 BuiltinDocumentationLink {
 label: "isobject",
 target: BuiltinDocumentationLinkTarget::Documentation("./isobject") }
]
,

    media: &[
]
,

    evidence: BuiltinDocumentationEvidence {
 implementation: &[
BuiltinDocumentationLink {
 label: "Runtime implementation",
 target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/isvalid.rs") }
]
,
 verification: &[
BuiltinEvidenceReference {
 kind: BuiltinEvidenceKind::UnitTest,
 label: "Handle and invalid input behavior",
 location: "builtins::introspection::isvalid::tests" }
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
