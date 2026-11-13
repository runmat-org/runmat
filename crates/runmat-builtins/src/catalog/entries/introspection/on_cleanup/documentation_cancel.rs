use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("cancel"), slug: Some("cancel"),
    summary: "Deactivate cleanup or request cancellation of an execution handle.",
    description: "`cancel(target)` deactivates an onCleanup callback without invoking it, or asks the selected execution service to cancel a future, task, job, or supported parallel future aggregate. `status = cancel(target)` returns host double scalar zero after the request succeeds. This status-returning lifecycle/typed-handle form is a RunMat contract, not a claim of identical MATLAB signatures.",
    keywords: &["cancel", "onCleanup", "future", "task", "job", "parfeval"],
    related: &["onCleanup", "delete", "parfeval", "parfevalOnAll", "fetchOutputs"],
    sections: &[
        BuiltinDocumentationSection { heading: "Supported targets", paragraphs: &[
            "An onCleanup handle is deactivated without executing its callback. Its aliases remain handles; later delete or final release does not replay the cancelled callback. An already invalid onCleanup handle is a successful no-op.",
            "Typed Future, Task, and Job values are forwarded unchanged to the selected runtime execution service. Scope, generation, supported backend, and cancellation state are validated by that owner, not reconstructed from numbers or class-name strings.",
            "parallel.FevalFuture objects and arrays created by parfeval carry actual task handles. parallel.FevalOnAllFuture carries the corresponding task aggregate. Every stored task representation is validated before any cancellation is requested, then tasks are visited in their existing order without a second task workspace. Scope and generation validation remain with the execution service for each request.",
            "Cancellation requests can succeed after work has already completed. A zero status confirms that the selected service accepted the request; it does not promise that work never ran or that a remote job has synchronously stopped. A service failure stops traversal and retains its diagnostic; requests already accepted for earlier tasks are not rolled back.",
        ] },
        BuiltinDocumentationSection { heading: "Outputs, errors, and placement", paragraphs: &[
            "Zero or one output is supported. Requesting more than one output fails before deactivation or execution-service calls. Automatic display is suppressed; an explicitly requested status is always scalar double zero.",
            "Numeric values, including integer IDs and resident numeric arrays, are not handles. Unsupported target values reject without conversion or GPU gather. Execution targets require a selected runtime context; active cleanup targets also require their session lifecycle host. Service, scope, generation, and lifecycle failures propagate and never trigger GPU-gather retry.",
            "The binding is impure, host-side, a fusion boundary, and requires the process execution stack. This cancellation call does not await completion or invoke a cleanup callback. Native and Wasm bindings use their existing execution/lifecycle services; availability of a particular task or job backend is determined by that service. Handles cannot be transferred as independent distributed worker data.",
        ] },
    ],
    examples: &[
        BuiltinExample {
            id: "deactivate-cleanup", title: "Disable a cleanup callback",
            program: "cleanupObj = onCleanup(@() disp('cleanup')); status = cancel(cleanupObj); delete(cleanupObj);",
            display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isa(status,'double')); assert(isscalar(status)); assert(status == 0);" },
        },
        BuiltinExample {
            id: "cancel-parfeval", title: "Request cancellation of scheduled work",
            program: "future = parfeval(@(x) x + 1, 1, 4); status = cancel(future);",
            display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Native, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isa(status,'double')); assert(status == 0); delete(gcp('nocreate'));" },
        },
        BuiltinExample {
            id: "reject-numeric-id", title: "Do not interpret integers as task IDs",
            program: "cancel(uint64(1));",
            display_output: None, compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::ExpectedError { identifier: "RunMat:onCleanup:InvalidObject" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does cancelling cleanup invoke its callback?", answer: "No. Use delete(cleanupObj) to execute an active callback; cancel deactivates it without invocation." },
        BuiltinDocumentationFaq { question: "Can I cancel using an integer task number?", answer: "No. Pass the actual lifecycle or execution handle, preserving the owning scope and generation." },
        BuiltinDocumentationFaq { question: "Does zero mean the task never ran?", answer: "No. It means the cancellation request succeeded; completion and cancellation races belong to the execution service." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "onCleanup", target: BuiltinDocumentationLinkTarget::Builtin("onCleanup") },
        BuiltinDocumentationLink { label: "parfeval", target: BuiltinDocumentationLinkTarget::Builtin("parfeval") },
        BuiltinDocumentationLink { label: "parfevalOnAll", target: BuiltinDocumentationLinkTarget::Builtin("parfevalOnAll") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Selected cancellation and lifecycle operations", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/on_cleanup.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed cancellation, selected contexts, output preflight, and cleanup lifecycle", location: "crates/runmat-runtime/src/builtins/introspection/on_cleanup.rs::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Canonical operation facts and executable metadata", location: "crates/runmat-builtins/src/catalog/entries/introspection/on_cleanup/tests.rs" },
        ],
        notes: &["The operation reuses existing execution-service and session lifecycle ownership; it introduces no callback executor or parallel registry."],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
