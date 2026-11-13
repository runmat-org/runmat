use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("addlistener"), slug: Some("addlistener"),
    summary: "Register a callback on a handle-object event.",
    description: "`listener = addlistener(target,event_name,callback)` registers a callback with the current runtime event registry and returns a listener value. Registration occurs even when the result is discarded.",
    keywords: &["events", "listener", "notify", "handle", "classdef"],
    related: &["notify", "isvalid"],
    sections: &[
        BuiltinDocumentationSection { heading: "Receiver and callback", paragraphs: &[
            "The receiver must be a live handle object. Event names are case-sensitive text. The callback may be a bound function handle, closure, supported named handle, or textual function reference. Text references are resolved through the current semantic function catalog when possible; callback invocation occurs later in notify.",
            "The event registry retains explicit GC roots for target and callback until registration invalidation. The current implementation does not verify that the class declared the event or that the callback signature matches at registration time.",
            "Numeric or GPU-resident values are not direct target, event-name, or callback controls. Values captured by a callback remain under that callback's ordinary ownership rules and are not gathered by addlistener.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and output", paragraphs: &[
            "Native and browser hosts use the context-owned event registry. Registration is a host mutation and a fusion boundary, not a GPU kernel or automatically replayable operation. A worker's independent object registry does not inherit the originating listener.",
            "At most one output is available; a zero-output statement still registers the listener. Current listener invalidation and target lifetime are managed by the runtime's listener/GC services.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "register-listener", title: "Register an event callback",
        program: "h = new_handle_object('EventTarget'); listener = addlistener(h, 'Changed', @disp);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isvalid(listener)); assert(isa(listener,'listener'));",
        },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Does registration invoke the callback?",
        answer: "No. notify invokes enabled, valid listeners in registry order; registration only retains the target and callback.",
    }],
    links: &[
        BuiltinDocumentationLink { label: "notify", target: BuiltinDocumentationLinkTarget::Builtin("notify") },
        BuiltinDocumentationLink { label: "Class events", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/docs/CLASSES.md#events") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime event registration", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/event_runtime.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Listener GC and semantic callback resolution", location: "crates/runmat-runtime/src/lib.rs::tests" }],
        notes: &["Native/browser callback and cancellation execution still require composed qualification."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
