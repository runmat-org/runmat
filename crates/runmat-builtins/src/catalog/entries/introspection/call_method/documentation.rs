use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("call_method"),
    slug: Some("call_method"),
    summary: "Dispatch an object method through the internal class protocol.",
    description: "`call_method(base, method, varargin)` is a hidden runtime helper. It validates an object receiver and nonempty method name, attempts class-member dispatch, then applies the callable fallback route when the member is undefined.",
    keywords: &["classdef", "method", "dispatch", "object"],
    related: &["builtin", "subsref"],
    sections: &[
        BuiltinDocumentationSection { heading: "Internal dispatch", paragraphs: &[
            "The receiver must be an object or handle object. The method name is trimmed and must remain nonempty. Additional arguments and the caller's requested output count are forwarded to the resolved method. Undefined external members may use callable fallback; other errors are retained.",
            "The resolved method owns output class, shape, residency, errors, effects, capabilities, and provider behavior. Dispatch can reenter user code, suspend, and require a process stack; unknown method identities cannot be inferred as host-only or pure.",
            "This is a HiddenInternal completion-policy binding, not a separate user-facing API. Object ownership and callback captures follow the shared execution root and class registry services.",
        ] },
    ],
    examples: &[],
    example_exemption: Some("Hidden method-dispatch helper; public object and indexing examples cover this route."),
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "builtin", target: BuiltinDocumentationLinkTarget::Builtin("builtin") },
        BuiltinDocumentationLink { label: "subsref", target: BuiltinDocumentationLinkTarget::Builtin("subsref") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime method dispatcher", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/call_method.rs") }],
        verification: &[],
        notes: &["No editable reference JSON sidecar existed for this hidden identity at the signed base. Composed method-dispatch tests remain required."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
