//! Queue array changes; physical payload effects occur during commit.

use super::super::super::{
    contract::TRANSACTION_QUEUE_EFFECTS, descriptor as d, documentation as docs,
    transaction_inference::TransactionForm, transaction_integer as integer,
};
use crate::*;

const VERIFY: &[BuiltinEvidenceReference] = &[BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Queue create, resize, fill, and delete before commit",
    location: "builtins::io::data::tests::tx_create_resize_fill_and_delete_array",
}];

const WRITE_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "DataTransaction.write",
    summary: "Queue a numeric slice write.",
    description: "`tf = DataTransaction.write(tx, arrayName, sliceSpec, values)` retains the supplied slice and values in an open transaction; commit applies the write to the named stored array.",
    keywords: &["DataTransaction.write", "transaction", "array", "slice"],
    related: &["DataArray.write", "DataTransaction.commit"],
    sections: &[BuiltinDocumentationSection { heading: "Validation and ownership", paragraphs: &["The queue call validates the managed transaction and array name, but slice geometry, declared dtype, and values are checked during commit. Values may be resident until the commit path gathers them; no GPU execution or publication is promised by this queue call."] }],
    examples: &[docs::example("tx-write", "Queue a slice write", "schema = struct(); schema.arrays.sample = struct('dtype', 'uint64', 'shape', [1 2]);\nds = data.create('tx-write.data', schema);\ntx = Dataset.begin(ds);\ntf = DataTransaction.write(tx, 'sample', {':', 2}, uint64(8));\nDataTransaction.commit(tx);\narr = Dataset.array(ds, 'sample');", "assert(tf); assert(isequal(DataArray.read(arr), uint64([0 8])));")],
    faqs: &[BuiltinDocumentationFaq { question: "Does queueing write array chunks?", answer: "No. Commit writes fresh array revisions. A failed publication can leave unreferenced revision files, but does not overwrite previously referenced chunks." }],
    verification: VERIFY,
});
transaction_entry!(
    DATATX_WRITE_CATALOG_ENTRY,
    DATATX_WRITE_DESCRIPTOR,
    infer_write,
    "DataTransaction.write",
    "tf = DataTransaction.write(tx, arrayName, sliceSpec, values)",
    &[d::TRANSACTION, d::NAME, d::SLICE_REQUIRED, d::VALUES],
    &[d::OUT_BOOL],
    TransactionForm::Write,
    WRITE_DOC,
    TRANSACTION_QUEUE_EFFECTS,
    BuiltinPurity::Impure,
    integer::WRITE,
    None
);

const RESIZE_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "DataTransaction.resize",
    summary: "Queue a destructive zero-filled array resize.",
    description: "`tf = DataTransaction.resize(tx, arrayName, newShape)` validates exact dimensions now and queues a resize; commit recreates zero-filled payload storage and updates the manifest shape.",
    keywords: &["DataTransaction.resize", "transaction", "shape", "dataset"],
    related: &["DataArray.resize", "DataTransaction.commit"],
    sections: &[BuiltinDocumentationSection { heading: "Old values", paragraphs: &["Commit does not preserve previous array values. Shape dimensions must be nonnegative exact integers within checked platform bounds; trailing options are not accepted."] }],
    examples: &[docs::example("tx-resize", "Queue a resize", "schema = struct(); schema.arrays.sample = struct('dtype', 'f64', 'shape', [1 1]);\nds = data.create('tx-resize.data', schema);\ntx = Dataset.begin(ds);\ntf = DataTransaction.resize(tx, 'sample', [1 2]);\nDataTransaction.commit(tx);\narr = Dataset.array(ds, 'sample');", "assert(tf); assert(isequal(DataArray.shape(arr), [1 2])); assert(all(DataArray.read(arr) == 0));")],
    faqs: &[BuiltinDocumentationFaq { question: "Is previous content copied?", answer: "No. Commit creates zero-filled storage at the new shape." }],
    verification: VERIFY,
});
transaction_entry!(
    DATATX_RESIZE_CATALOG_ENTRY,
    DATATX_RESIZE_DESCRIPTOR,
    infer_resize,
    "DataTransaction.resize",
    "tf = DataTransaction.resize(tx, arrayName, newShape)",
    &[d::TRANSACTION, d::NAME, d::NEW_SHAPE],
    &[d::OUT_BOOL],
    TransactionForm::Resize,
    RESIZE_DOC,
    TRANSACTION_QUEUE_EFFECTS,
    BuiltinPurity::Impure,
    integer::RESIZE,
    None
);

const FILL_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "DataTransaction.fill",
    summary: "Queue a scalar fill for a stored array or slice.",
    description: "`tf = DataTransaction.fill(tx, arrayName, value, sliceSpec)` stores a pending scalar fill; `sliceSpec` is optional and the first trailing argument becomes the slice.",
    keywords: &["DataTransaction.fill", "transaction", "fill", "slice"],
    related: &["DataArray.fill", "DataTransaction.commit"],
    sections: &[BuiltinDocumentationSection { heading: "Deferred validation", paragraphs: &["The queue retains the original value and optional slice. Commit validates scalar shape, applies declared-dtype casting, and writes chunks. At most one optional slice is accepted."] }],
    examples: &[docs::example("tx-fill", "Queue a typed fill", "schema = struct(); schema.arrays.sample = struct('dtype', 'uint64', 'shape', [1 2]);\nds = data.create('tx-fill.data', schema);\ntx = Dataset.begin(ds);\ntf = DataTransaction.fill(tx, 'sample', uint64(5));\nDataTransaction.commit(tx);\narr = Dataset.array(ds, 'sample');", "assert(tf); assert(isequal(DataArray.read(arr), uint64([5 5])));")],
    faqs: &[BuiltinDocumentationFaq { question: "Does the fill run immediately?", answer: "No. It is replayed at commit." }],
    verification: VERIFY,
});
transaction_entry!(
    DATATX_FILL_CATALOG_ENTRY,
    DATATX_FILL_DESCRIPTOR,
    infer_fill,
    "DataTransaction.fill",
    "tf = DataTransaction.fill(tx, arrayName, value, sliceSpec)",
    &[d::TRANSACTION, d::NAME, d::VALUE, d::SLICE_OPTIONAL],
    &[d::OUT_BOOL],
    TransactionForm::Fill,
    FILL_DOC,
    TRANSACTION_QUEUE_EFFECTS,
    BuiltinPurity::Impure,
    integer::FILL,
    None
);

const DELETE_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "DataTransaction.delete_array",
    summary: "Queue deletion of a named stored array.",
    description: "`tf = DataTransaction.delete_array(tx, arrayName)` queues the name; commit removes its entry from the published manifest through the pinned provider. Stored revisions remain available to snapshots and readers that still reference them.",
    keywords: &["DataTransaction.delete_array", "transaction", "delete", "dataset"],
    related: &["DataTransaction.create_array", "DataTransaction.commit"],
    sections: &[BuiltinDocumentationSection { heading: "Error timing", paragraphs: &["A missing stored array or manifest publication failure is reported during commit, not during queue insertion. Rejected publication leaves the array and its stored revisions unchanged."] }],
    examples: &[docs::example("tx-delete-array", "Queue an array deletion", "schema = struct(); schema.arrays.sample = struct('dtype', 'f64', 'shape', [1 1]);\nds = data.create('tx-delete.data', schema);\ntx = Dataset.begin(ds);\ntf = DataTransaction.delete_array(tx, 'sample');\nDataTransaction.commit(tx);\nfresh = Dataset.refresh(ds);", "assert(tf); assert(~Dataset.has_array(fresh, 'sample'));")],
    faqs: &[BuiltinDocumentationFaq { question: "Can abort restore a committed deletion?", answer: "No. Abort only discards pending changes before commit." }],
    verification: VERIFY,
});
transaction_entry!(
    DATATX_DELETE_ARRAY_CATALOG_ENTRY,
    DATATX_DELETE_ARRAY_DESCRIPTOR,
    infer_delete,
    "DataTransaction.delete_array",
    "tf = DataTransaction.delete_array(tx, arrayName)",
    &[d::TRANSACTION, d::NAME],
    &[d::OUT_BOOL],
    TransactionForm::DeleteArray,
    DELETE_DOC,
    TRANSACTION_QUEUE_EFFECTS,
    BuiltinPurity::Impure,
    &[],
    Some(&integer::LIFECYCLE_AUDIT)
);

const CREATE_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "DataTransaction.create_array",
    summary: "Queue creation of a typed stored array.",
    description: "`tf = DataTransaction.create_array(tx, arrayName, meta)` validates array metadata and queues creation; commit publishes the array in the manifest and writes initial storage.",
    keywords: &["DataTransaction.create_array", "transaction", "schema", "dataset"],
    related: &["DataTransaction.delete_array", "DataTransaction.commit"],
    sections: &[BuiltinDocumentationSection { heading: "Metadata", paragraphs: &["`meta` is a structure with a required shape, optional dtype and chunk shape, and optional codec. Shape and chunk dimensions are validated when queued; an existing array name conflicts at commit."] }],
    examples: &[docs::example("tx-create-array", "Queue a new typed array", "schema = struct(); schema.arrays = struct();\nds = data.create('tx-create.data', schema);\ntx = Dataset.begin(ds);\nmeta = struct('dtype', 'uint64', 'shape', [1 2], 'chunk', [1 2]);\ntf = DataTransaction.create_array(tx, 'sample', meta);\nDataTransaction.commit(tx);\nfresh = Dataset.refresh(ds);\narr = Dataset.array(fresh, 'sample');", "assert(tf); assert(strcmp(DataArray.dtype(arr), 'uint64')); assert(isequal(DataArray.shape(arr), [1 2]));")],
    faqs: &[BuiltinDocumentationFaq { question: "Can I create a duplicate name?", answer: "No. Commit rejects a name already in the manifest." }],
    verification: VERIFY,
});
transaction_entry!(
    DATATX_CREATE_ARRAY_CATALOG_ENTRY,
    DATATX_CREATE_ARRAY_DESCRIPTOR,
    infer_create,
    "DataTransaction.create_array",
    "tf = DataTransaction.create_array(tx, arrayName, meta)",
    &[d::TRANSACTION, d::NAME, d::META],
    &[d::OUT_BOOL],
    TransactionForm::CreateArray,
    CREATE_DOC,
    TRANSACTION_QUEUE_EFFECTS,
    BuiltinPurity::Impure,
    integer::CREATE_ARRAY,
    None
);
