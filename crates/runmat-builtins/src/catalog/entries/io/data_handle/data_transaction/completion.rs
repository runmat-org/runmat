//! Transaction commit, alias, and abort contracts.

use super::super::{
    contract::{TRANSACTION_COMMIT_EFFECTS, TRANSACTION_QUEUE_EFFECTS},
    descriptor as d, documentation as docs,
    transaction_inference::TransactionForm,
    transaction_integer::LIFECYCLE_AUDIT,
};
use crate::*;

const VERIFY: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Pinned transaction owner survives ambient provider replacement", location: "builtins::io::data::tests::data_create_retains_selected_provider_and_relative_cwd_after_first_write" },
    BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Queued create/resize/fill/delete and commit", location: "builtins::io::data::tests::tx_create_resize_fill_and_delete_array" },
];

const COMMIT_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "DataTransaction.commit",
    summary: "Replay queued changes through the transaction's pinned filesystem owner.",
    description: "`tf = DataTransaction.commit(tx, options)` checks the captured manifest sequence, optionally checks `options.if_manifest`, applies queued array and attribute changes, writes the manifest, and removes the transaction on success.",
    keywords: &["DataTransaction.commit", "transaction", "manifest", "dataset"],
    related: &["Dataset.begin", "DataTransaction.abort", "commit"],
    sections: &[BuiltinDocumentationSection { heading: "Failure and concurrency", paragraphs: &["Manifest sequence conflicts reject before queued changes. Array mutations write fresh revisions and publish their references through the manifest. A rejected manifest publication leaves previously referenced payloads, chunk indexes and chunks unchanged. Failed writes can leave unreferenced revision files; they are retained because snapshots and readers can still reference older revisions. Commit is not a filesystem-wide transaction. A failed transaction remains open unless explicitly aborted."] }],
    examples: &[docs::example("tx-commit", "Commit an attribute update", "schema = struct(); schema.arrays = struct();\nds = data.create('tx-commit.data', schema);\ntx = Dataset.begin(ds);\nDataTransaction.set_attr(tx, 'stage', 'ready');\ntf = DataTransaction.commit(tx);\nfresh = Dataset.refresh(ds);", "assert(tf); assert(strcmp(Dataset.get_attr(fresh, 'stage'), 'ready'));")],
    faqs: &[BuiltinDocumentationFaq { question: "Does commit switch to the current filesystem provider?", answer: "No. It uses the exact provider and root captured by the managed transaction." }],
    verification: VERIFY,
});
transaction_entry!(
    DATATX_COMMIT_CATALOG_ENTRY,
    DATATX_COMMIT_DESCRIPTOR,
    infer_commit,
    "DataTransaction.commit",
    "tf = DataTransaction.commit(tx, options)",
    &[d::TRANSACTION, d::COMMIT_OPTIONS],
    &[d::OUT_BOOL],
    TransactionForm::Commit,
    COMMIT_DOC,
    TRANSACTION_COMMIT_EFFECTS,
    BuiltinPurity::Impure,
    &[],
    Some(&LIFECYCLE_AUDIT),
    BuiltinAsyncBehavior::MaySuspend
);

const ALIAS_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "commit",
    summary: "Commit a managed DataTransaction through the standalone alias.",
    description: "`tf = commit(tx, options)` delegates to `DataTransaction.commit` for a managed DataTransaction receiver and has the same manifest precondition and pinned-owner behavior.",
    keywords: &["commit", "DataTransaction", "dataset", "manifest"],
    related: &["DataTransaction.commit", "DataTransaction.abort"],
    sections: &[BuiltinDocumentationSection { heading: "Receiver and failure", paragraphs: &["Other objects are rejected. Array changes use fresh revisions; failure before manifest publication leaves previously published array objects unchanged and can leave unreferenced files. This alias does not replay the operation on another provider."] }],
    examples: &[docs::example("tx-commit-alias", "Use the standalone commit alias", "schema = struct(); schema.arrays = struct();\nds = data.create('tx-alias.data', schema);\ntx = Dataset.begin(ds);\nDataTransaction.set_attr(tx, 'stage', 'ready');\ntf = commit(tx);\nfresh = Dataset.refresh(ds);", "assert(tf); assert(strcmp(Dataset.get_attr(fresh, 'stage'), 'ready'));")],
    faqs: &[BuiltinDocumentationFaq { question: "Is this a different transaction mechanism?", answer: "No. It is an alias of DataTransaction.commit and requires the same managed receiver." }],
    verification: VERIFY,
});
transaction_entry!(
    COMMIT_ALIAS_CATALOG_ENTRY,
    COMMIT_ALIAS_DESCRIPTOR,
    infer_alias,
    "commit",
    "tf = commit(tx, options)",
    &[d::TRANSACTION, d::COMMIT_OPTIONS],
    &[d::OUT_BOOL],
    TransactionForm::Commit,
    ALIAS_DOC,
    TRANSACTION_COMMIT_EFFECTS,
    BuiltinPurity::Impure,
    &[],
    Some(&LIFECYCLE_AUDIT),
    BuiltinAsyncBehavior::MaySuspend
);

const ABORT_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "DataTransaction.abort",
    summary: "Discard a process-local pending transaction.",
    description: "`tf = DataTransaction.abort(tx)` marks an open transaction aborted and removes it from the process-local registry without replaying queued writes.",
    keywords: &["DataTransaction.abort", "transaction", "discard", "dataset"],
    related: &["DataTransaction.commit", "DataTransaction.status"],
    sections: &[BuiltinDocumentationSection { heading: "Scope", paragraphs: &["Abort drops queued changes only. It cannot undo effects of a previously failed commit that reached filesystem writes."] }],
    examples: &[docs::example("tx-abort", "Discard a queued attribute", "schema = struct(); schema.arrays = struct();\nds = data.create('tx-abort.data', schema);\ntx = Dataset.begin(ds);\nDataTransaction.set_attr(tx, 'stage', 'discard');\ntf = DataTransaction.abort(tx);\nfresh = Dataset.refresh(ds);", "assert(tf); assert(Dataset.get_attr(fresh, 'stage', 0) == 0);")],
    faqs: &[BuiltinDocumentationFaq { question: "Does abort delete array chunks?", answer: "No. It discards the queue; it does not write dataset files." }],
    verification: VERIFY,
});
transaction_entry!(
    DATATX_ABORT_CATALOG_ENTRY,
    DATATX_ABORT_DESCRIPTOR,
    infer_abort,
    "DataTransaction.abort",
    "tf = DataTransaction.abort(tx)",
    &[d::TRANSACTION],
    &[d::OUT_BOOL],
    TransactionForm::Abort,
    ABORT_DOC,
    TRANSACTION_QUEUE_EFFECTS,
    BuiltinPurity::Impure,
    &[],
    Some(&LIFECYCLE_AUDIT)
);
