use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("make_anon"),
    slug: Some("make_anon"),
    summary: "Create the internal anonymous-function text representation.",
    description: "Formats exactly @anon(params) body as a scalar string. It does not parse the parameter list or body, create a closure, capture a workspace, or execute the text.",
    keywords: &[
        "classdef",
        "internal"
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Execution and resource boundary",
            paragraphs: &[
                "The canonical contextual binding retains its original BuiltinArguments for the complete call and scopes the selected RuntimeContext around the existing semantic helper. It accepts 2 inputs and at most one output. Text inputs preserve existing scalar string/one-row character conversion, including legacy scalar integer decimal, double and logical formatting. Integer arrays and provider values are not gathered.",
                "Construction and registry backing do not yet have an exact finite-resource certificate. Cancellation and revoked allocation authority are checked before text projection or registry/GC effects; managed finite-resource invocations are refused with RunMat:Placement:InsufficientResourceEvidence. There is no accelerator provider, implicit gather, distributed fallback, or fusion region."
            ]
        }
    ],
    example_exemption: Some("Hidden internal executable helper; class lowering and runtime fixture tests exercise this identity rather than presenting it as a public API."),
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "Runtime helper",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/make_anon.rs")
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
