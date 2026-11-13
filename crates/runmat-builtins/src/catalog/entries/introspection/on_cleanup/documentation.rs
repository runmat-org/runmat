use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("onCleanup"),
    slug: Some("onCleanup"),
    summary: "Run a function once when a cleanup object's last reachable owner is released.",
    description: "`cleanupObj = onCleanup(cleanupFun)` creates a host handle object that invokes a zero-input function handle or closure on explicit deletion or final owner release. It is useful for restoring state, closing files, removing temporary folders, and other cleanup tasks.",
    keywords: &["onCleanup", "cleanup", "delete", "cancel", "scope", "resource"],
    related: &["clear", "clearvars", "delete", "fopen"],
    sections: &[
        BuiltinDocumentationSection { heading: "Callback and ownership", paragraphs: &[
            "The callback receives zero input arguments and has zero requested outputs. Use an anonymous function such as `@() fclose(fid)` to capture arguments. Numeric values, including exact integers and GPU-resident numeric values, are not callbacks; captured values retain their ordinary ownership and are not inspected or gathered by onCleanup.",
            "The callback runs at most once. Clearing one workspace name does not invoke it while another alias, active capture, pending operation, or external root still owns the object. `clear`, `clearvars`, and `clear cleanupObj` detach selected workspace bindings first and dispatch an eligible callback after the last owner is released.",
            "`delete(cleanupObj)` invokes the onCleanup handle method immediately and deactivates the object. `cancel(cleanupObj)` deactivates it without running the callback. Repeated delete, clear, or clearvars operations do not replay a deactivated callback.",
        ] },
        BuiltinDocumentationSection { heading: "Host execution and limits", paragraphs: &[
            "The session lifecycle host owns callback dispatch. Construction requires that host; a bare runtime context rejects a valid callback before object admission. A normal hosted session drains admitted callbacks before graceful close. Forced disposal may abandon a queued callback but does not replay an already admitted one.",
            "The cleanup object is a host lifecycle handle and a fusion boundary. Construction launches no GPU kernel. Callback bodies may call GPU-aware functions, but their own contracts determine residency and device work. Handle ownership and callback execution are session-local, not transferable to an independent distributed worker.",
            "The callback may mutate workspace or environment state, throw, or suspend when it is eventually invoked. Those effects belong to callback dispatch or an explicit delete/clear path, not to construction's synchronous registration. A discarded construction result can become eligible immediately after its output lease retires.",
        ] },
    ],
    examples: &[
        BuiltinExample {
            id: "restore-folder", title: "Restore a folder after final release",
            program: "old = pwd; cleanupObj = onCleanup(@() cd(old)); cd(tempdir); clear cleanupObj; restored = strcmp(pwd,old);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(restored);" },
        },
        BuiltinExample {
            id: "close-file", title: "Close a file when the handle is released",
            program: "fid = fopen('data.txt','w'); cleanupObj = onCleanup(@() fclose(fid)); fprintf(fid,'sample\\n'); delete(cleanupObj);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(fid > 0);" },
        },
        BuiltinExample {
            id: "cancel", title: "Disable an unneeded cleanup",
            program: "cleanupObj = onCleanup(@() disp('cleanup')); status = cancel(cleanupObj); delete(cleanupObj);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(status == 0);" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Can I pass arguments directly to the cleanup function?", answer: "No. Use an anonymous function that captures them, for example `onCleanup(@() fclose(fid))`." },
        BuiltinDocumentationFaq { question: "Does repeated delete invoke the callback again?", answer: "No. The object is deactivated before callback invocation, so later deletion or release does not replay it." },
        BuiltinDocumentationFaq { question: "Does cancel delete the object?", answer: "No. It deactivates the callback; the handle may still exist." },
        BuiltinDocumentationFaq { question: "Will clearing one name run cleanup if another name holds the object?", answer: "No. Cleanup waits for the final reachable owner unless the object is explicitly deleted." },
        BuiltinDocumentationFaq { question: "Does GPU residency matter?", answer: "The cleanup handle is host-owned. Device behavior of a callback is determined by that callback's operations." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "clear", target: BuiltinDocumentationLinkTarget::Builtin("clear") },
        BuiltinDocumentationLink { label: "clearvars", target: BuiltinDocumentationLinkTarget::Builtin("clearvars") },
        BuiltinDocumentationLink { label: "delete", target: BuiltinDocumentationLinkTarget::Builtin("delete") },
        BuiltinDocumentationLink { label: "fopen", target: BuiltinDocumentationLinkTarget::Builtin("fopen") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "onCleanup producer and handle methods", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/on_cleanup.rs") },
            BuiltinDocumentationLink { label: "Session lifecycle registry", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/on_cleanup/lifecycle/registry.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Producer lease and explicit delete/cancel", location: "crates/runmat-runtime/src/builtins/introspection/on_cleanup.rs::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Workspace clear and clearvars lifecycle", location: "crates/runmat-runtime/src/builtins/introspection/clear.rs; crates/runmat-runtime/src/builtins/introspection/clearvars.rs" },
        ],
        notes: &["Native/browser composed lifecycle and observer gates remain subject to integration verification."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
