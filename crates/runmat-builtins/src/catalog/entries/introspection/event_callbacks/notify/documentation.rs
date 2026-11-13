use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("notify"), slug: Some("notify"),
    summary: "Dispatch a handle-object event to registered callbacks.",
    description: "`notify(target,event_name)` calls enabled listeners for a live handle object. An event-data value, or RunMat's broader callback-argument list, is forwarded without numeric conversion. The current runtime returns double zero on success when a result is requested.",
    keywords: &["notify", "events", "listeners", "handle"],
    related: &["addlistener"],
    sections: &[
        BuiltinDocumentationSection { heading: "Dispatch", paragraphs: &[
            "The target must be a live handle object and the event name is case-sensitive text. The documented data form is an event.EventData subclass object; the runtime also accepts multiple callback arguments as a RunMat extension, rejected when extensions are disabled. Nested integer or resident values are passed as ordinary callback payloads, not converted or gathered by notify.",
            "Listeners are snapshotted from the context-owned registry, and the receiver, callbacks, and nested event-data/extra arguments are rooted through dispatch. Listeners run sequentially through the semantic feval path with zero requested callback outputs. A callback can mutate workspace, objects, graphics, files, or external state, suspend, throw, or reenter notify. Those effects and capabilities are conservatively dynamic when registrations are not known statically.",
            "A callback error propagates with its original cause. Earlier callbacks may already have executed, so failure or cancellation cannot authorize replay of notify. The runtime does not promise atomic rollback of callback effects.",
        ] },
        BuiltinDocumentationSection { heading: "Placement", paragraphs: &[
            "Dispatch is a host-side fusion boundary and requires the process execution stack for callback reentry. Native and browser hosts share the semantic contract; worker/distributed transfer does not transfer the originating object's listener registry. No numeric provider kernel or implicit GPU materialization is performed by notify.",
            "A statement with no requested output still dispatches. At most one optional status result is returned by RunMat; an explicitly requested status is an extension rejected when extensions are disabled, and is not a MATLAB event-data output.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "dispatch-event", title: "Notify a registered listener",
        program: "h = new_handle_object('EventTarget'); listener = addlistener(h, 'Changed', @disp); status = notify(h, 'Changed');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(status == 0); assert(isvalid(listener));",
        },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Can notify be retried after a callback error?",
        answer: "No. Prior callbacks may already have run; the error is after-effect or conservatively unphased, never a proof that replay is safe.",
    }],
    links: &[BuiltinDocumentationLink { label: "addlistener", target: BuiltinDocumentationLinkTarget::Builtin("addlistener") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime event dispatch", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/event_runtime.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Callback identity and target rejection", location: "crates/runmat-runtime/src/lib.rs::tests" }],
        notes: &["Callback execution, browser reentry, and cancellation after partial dispatch require composed tests."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
