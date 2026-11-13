//! Transaction identity and process-local status inspection.

use super::super::{
    contract::TRANSACTION_READ_EFFECTS, descriptor as d, documentation as docs,
    transaction_inference::TransactionForm, transaction_integer::LIFECYCLE_AUDIT,
};
use crate::*;

const VERIFY: &[BuiltinEvidenceReference] = &[BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Managed transaction status and exact pinned owner",
    location: "builtins::io::data::tests::data_create_retains_selected_provider_and_relative_cwd_after_first_write",
}];

const ID_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "DataTransaction.id",
    summary: "Return the transaction's process-local identifier.",
    description: "`id = DataTransaction.id(tx)` reads the identifier from a live managed transaction handle. The identifier is presentation, not authority to construct another transaction.",
    keywords: &["DataTransaction.id", "transaction", "identifier", "dataset"],
    related: &["Dataset.begin", "DataTransaction.status"],
    sections: &[BuiltinDocumentationSection { heading: "Lifetime", paragraphs: &["Commit and abort remove the process-local transaction. A stale or synthetic receiver fails managed-handle validation."] }],
    examples: &[docs::example("tx-id", "Inspect a transaction ID", "schema = struct(); schema.arrays = struct();\nds = data.create('tx-id.data', schema);\ntx = Dataset.begin(ds);\nid = DataTransaction.id(tx);", "assert(~isempty(id)); assert(DataTransaction.abort(tx));")],
    faqs: &[BuiltinDocumentationFaq { question: "Can the ID reopen a transaction?", answer: "No. The managed handle carries sealed owner authority; the text identifier alone cannot reopen it." }],
    verification: VERIFY,
});
transaction_entry!(
    DATATX_ID_CATALOG_ENTRY,
    DATATX_ID_DESCRIPTOR,
    infer_id,
    "DataTransaction.id",
    "id = DataTransaction.id(tx)",
    &[d::TRANSACTION],
    &[d::OUT_STRING],
    TransactionForm::Id,
    ID_DOC,
    TRANSACTION_READ_EFFECTS,
    BuiltinPurity::DeterministicReadOnly,
    &[],
    Some(&LIFECYCLE_AUDIT)
);

const STATUS_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "DataTransaction.status",
    summary: "Report an open transaction's process-local state.",
    description: "`status = DataTransaction.status(tx)` returns `open` while the managed transaction remains in the process-local registry.",
    keywords: &["DataTransaction.status", "transaction", "state", "dataset"],
    related: &["DataTransaction.commit", "DataTransaction.abort"],
    sections: &[BuiltinDocumentationSection { heading: "Removal", paragraphs: &["Successful commit and abort remove the registry entry; a subsequent status call on that handle reports transaction-not-found rather than a durable committed or aborted status."] }],
    examples: &[docs::example("tx-status", "Inspect and abort an open transaction", "schema = struct(); schema.arrays = struct();\nds = data.create('tx-status.data', schema);\ntx = Dataset.begin(ds);\nstatus = DataTransaction.status(tx);", "assert(strcmp(status, 'open')); assert(DataTransaction.abort(tx));")],
    faqs: &[BuiltinDocumentationFaq { question: "Is status persisted in the dataset?", answer: "No. It describes a process-local pending transaction." }],
    verification: VERIFY,
});
transaction_entry!(
    DATATX_STATUS_CATALOG_ENTRY,
    DATATX_STATUS_DESCRIPTOR,
    infer_status,
    "DataTransaction.status",
    "status = DataTransaction.status(tx)",
    &[d::TRANSACTION],
    &[d::OUT_STRING],
    TransactionForm::Status,
    STATUS_DOC,
    TRANSACTION_READ_EFFECTS,
    BuiltinPurity::DeterministicReadOnly,
    &[],
    Some(&LIFECYCLE_AUDIT)
);
