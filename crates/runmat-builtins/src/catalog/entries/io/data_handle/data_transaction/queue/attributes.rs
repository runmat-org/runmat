//! Queue attribute updates without publishing manifest changes.

use super::super::super::{
    contract::TRANSACTION_QUEUE_EFFECTS, descriptor as d, documentation as docs,
    transaction_inference::TransactionForm, transaction_integer as integer,
};
use crate::*;

const VERIFY: &[BuiltinEvidenceReference] = &[BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Queued attributes retain exact owner and integer values through commit",
    location: "builtins::io::data::tests::data_create_retains_selected_provider_and_relative_cwd_after_first_write",
}];

const SET_ATTR_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "DataTransaction.set_attr",
    summary: "Queue one manifest attribute update.",
    description: "`tf = DataTransaction.set_attr(tx, key, value)` records a key/value pair in an open process-local transaction; commit writes it to the pinned dataset's manifest.",
    keywords: &["DataTransaction.set_attr", "transaction", "attribute", "dataset"],
    related: &["DataTransaction.set_attrs", "DataTransaction.commit"],
    sections: &[BuiltinDocumentationSection { heading: "Value representation", paragraphs: &[docs::ATTRIBUTE_VALUE_BOUNDARY, "Scalar integer values retain exact mathematical value through JSON serialization, including full-width uint64. The original narrow integer storage class is not represented by manifest JSON. No filesystem write occurs at queue time; the value boundary is checked when commit serializes pending attributes."] }],
    examples: &[docs::example("tx-set-attr", "Queue and commit an attribute", "schema = struct(); schema.arrays = struct();\nds = data.create('tx-attr.data', schema);\ntx = Dataset.begin(ds);\ntf = DataTransaction.set_attr(tx, 'count', uint64(7));\nDataTransaction.commit(tx);\nfresh = Dataset.refresh(ds);", "assert(tf); assert(Dataset.get_attr(fresh, 'count') == uint64(7));")],
    faqs: &[BuiltinDocumentationFaq { question: "Is the update visible before commit?", answer: "No. It is held in the process-local transaction queue until commit succeeds." }],
    verification: VERIFY,
});
transaction_entry!(
    DATATX_SET_ATTR_CATALOG_ENTRY,
    DATATX_SET_ATTR_DESCRIPTOR,
    infer_set_attr,
    "DataTransaction.set_attr",
    "tf = DataTransaction.set_attr(tx, key, value)",
    &[d::TRANSACTION, d::KEY, d::VALUE],
    &[d::OUT_BOOL],
    TransactionForm::SetAttr,
    SET_ATTR_DOC,
    TRANSACTION_QUEUE_EFFECTS,
    BuiltinPurity::Impure,
    integer::SET_ATTR,
    None
);

const SET_ATTRS_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "DataTransaction.set_attrs",
    summary: "Queue all fields of an attribute structure.",
    description: "`tf = DataTransaction.set_attrs(tx, attrs)` copies structure fields into the open transaction's pending attributes; later fields with the same key replace earlier pending fields.",
    keywords: &["DataTransaction.set_attrs", "transaction", "attributes", "struct"],
    related: &["DataTransaction.set_attr", "DataTransaction.commit"],
    sections: &[BuiltinDocumentationSection { heading: "Commit boundary", paragraphs: &[docs::ATTRIBUTE_VALUE_BOUNDARY, "The call mutates only the process-local queue. Commit checks and serializes the pending field values to manifest JSON; a failed commit may leave previous chunk effects and the queue remains open."] }],
    examples: &[docs::example("tx-set-attrs", "Queue an attribute structure", "schema = struct(); schema.arrays = struct();\nds = data.create('tx-attrs.data', schema);\ntx = Dataset.begin(ds);\nattrs = struct('count', uint64(9), 'phase', 'ready');\ntf = DataTransaction.set_attrs(tx, attrs);\nDataTransaction.commit(tx);\nfresh = Dataset.refresh(ds);", "assert(tf); assert(Dataset.get_attr(fresh, 'count') == uint64(9)); assert(strcmp(Dataset.get_attr(fresh, 'phase'), 'ready'));")],
    faqs: &[BuiltinDocumentationFaq { question: "Can I pass a cell instead of a struct?", answer: "No. This form requires a structure." }],
    verification: VERIFY,
});
transaction_entry!(
    DATATX_SET_ATTRS_CATALOG_ENTRY,
    DATATX_SET_ATTRS_DESCRIPTOR,
    infer_set_attrs,
    "DataTransaction.set_attrs",
    "tf = DataTransaction.set_attrs(tx, attrs)",
    &[d::TRANSACTION, d::ATTRS],
    &[d::OUT_BOOL],
    TransactionForm::SetAttrs,
    SET_ATTRS_DOC,
    TRANSACTION_QUEUE_EFFECTS,
    BuiltinPurity::Impure,
    integer::SET_ATTRS,
    None
);
