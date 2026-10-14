use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("__runmat_call_bound_method__"),
    slug: Some("__runmat_call_bound_method__"),
    summary: "Invoke a bound method with its creator class access scope.",
    description: "`__runmat_call_bound_method__(base, method, scope, varargin)` is a hidden closure-invocation helper. It installs a valid creator-class scope for access checks, resolves a method against the receiver class and owner, and forwards the requested output count.",
    keywords: &["classdef", "method", "dispatch", "closure", "internal"],
    related: &["call_method", "builtin"],
    sections: &[
        BuiltinDocumentationSection { heading: "Internal bound-method route", paragraphs: &[
            "The receiver must resolve to an object class. The scope argument must be a string scalar; a trimmed empty string selects no creator-class guard. Nonempty valid class identities install an access-context guard for the delegated call. The method name is trimmed and must remain nonempty.",
            "The class registry chooses a declared method owner where available. The delegated external member receives the receiver, remaining arguments, and caller's requested output count. Output facts, provider residency, effects, and capabilities depend on that method, not on this wrapper.",
            "This binding retains HiddenInternal completion policy. It is a host dispatch and fusion boundary, but it does not gather resident values or impose a host-only result. Bound closure ownership and class-access context are runtime-local.",
        ] },
    ],
    examples: &[],
    example_exemption: Some("Hidden bound-method closure helper; class method call examples exercise this route."),
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "call_method", target: BuiltinDocumentationLinkTarget::Builtin("call_method") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime bound-method dispatcher", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/call_bound_method.rs") }],
        verification: &[],
        notes: &["No editable reference JSON sidecar existed for this hidden identity at the signed base. Composed method and class-access tests remain required."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
