use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("new_handle_object"),
    slug: Some("new_handle_object"),
    summary: "Create a handle object through class-default construction.",
    description: "Uses the existing class constructor service to initialize nonstatic inherited defaults. Existing handle-class instances are retained; value objects are allocated into a GC handle. Invalid class names, abstract classes, selected-owner mismatches and allocation failures retain their existing runtime errors. Unknown classes produce an empty class object before wrapping.",
    keywords: &[
        "classdef",
        "internal"
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Execution and resource boundary",
            paragraphs: &[
                "The canonical contextual binding retains its original BuiltinArguments for the complete call and scopes the selected RuntimeContext around the existing semantic helper. It accepts 1 inputs and at most one output. Text inputs preserve existing scalar string/one-row character conversion, including legacy scalar integer decimal, double and logical formatting. Integer arrays and provider values are not gathered.",
                "Construction and registry backing do not yet have an exact finite-resource certificate. Cancellation and revoked allocation authority are checked before text projection or registry/GC effects; managed finite-resource invocations are refused with RunMat:Placement:InsufficientResourceEvidence. There is no accelerator provider, implicit gather, distributed fallback, or fusion region."
            ]
        }
    ],
    example_exemption: Some("Hidden internal executable helper; class lowering and runtime fixture tests exercise this identity rather than presenting it as a public API."),
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Runtime helper",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/new_handle_object.rs")
            }
        ],
        verification: &[],
        notes: &[
            "First canonical documentation; no legacy reference JSON existed. Source-level contracts are not a compiled or finite-resource qualification receipt."
        ]
    },
    status: Some(BuiltinDocumentationStatus::Partial),
    ..BuiltinDocumentation::EMPTY
};
