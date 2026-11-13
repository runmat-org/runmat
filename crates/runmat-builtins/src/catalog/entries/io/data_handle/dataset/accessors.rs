//! Dataset identity, array listing, membership, and child-handle contracts.

use super::super::{
    contract::{self, Behavior, ACCESS_EFFECTS, FILESYSTEM, READ_EFFECTS},
    descriptor as d, documentation as docs,
    inference::{self, DatasetForm},
    integer::DATASET_NONATTRIBUTE_AUDIT,
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

const LOCAL: Behavior = Behavior {
    effects: ACCESS_EFFECTS,
    capabilities: &[],
    purity: BuiltinPurity::DeterministicReadOnly,
    async_behavior: BuiltinAsyncBehavior::NeverSuspends,
    integer_capabilities: &[],
    integer_audit: Some(&DATASET_NONATTRIBUTE_AUDIT),
};
const READ: Behavior = Behavior {
    effects: READ_EFFECTS,
    capabilities: FILESYSTEM,
    purity: BuiltinPurity::DeterministicReadOnly,
    async_behavior: BuiltinAsyncBehavior::MaySuspend,
    integer_capabilities: &[],
    integer_audit: Some(&DATASET_NONATTRIBUTE_AUDIT),
};

macro_rules! accessor {
    ($entry:ident, $descriptor:ident, $infer:ident, $name:literal, $label:literal, $inputs:expr, $output:expr, $form:expr, $doc:expr, $behavior:expr) => {
        d::receiver_descriptor!($descriptor, $label, $inputs, $output);
        fn $infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
            inference::dataset_infer(request, entry, $form)
        }
        pub const $entry: BuiltinCatalogEntry = contract::entry(
            BuiltinCatalogProvenance::new(file!(), module_path!()),
            $name,
            $doc,
            &$descriptor,
            $infer,
            $behavior,
        );
    };
}

const IDENTITY_VERIFICATION: &[BuiltinEvidenceReference] = &[BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Dataset identity and pinned-owner accessors",
    location: "builtins::io::data::tests::data_create_retains_selected_provider_and_relative_cwd_after_first_write",
}];

const PATH_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "Dataset.path",
    summary: "Return the Dataset handle's public path text.",
    description: "`path = Dataset.path(ds)` presents the path supplied when this managed Dataset handle was created or opened; the sealed resolved root, not this editable text property, authorizes later I/O.",
    keywords: &["Dataset.path", "dataset", "path"],
    related: &["data.open", "Dataset.id"],
    sections: &[BuiltinDocumentationSection { heading: "Handle identity", paragraphs: &["The result may be relative even though the handle privately retains an absolute root and exact filesystem provider. Moving a dataset does not rewrite an existing handle's presentation path."] }],
    examples: &[docs::example("dataset-path", "Read a created dataset path", "schema = struct(); schema.arrays = struct();\nds = data.create('path.data', schema);\np = Dataset.path(ds);", "assert(strcmp(p, 'path.data'));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Does changing the current directory retarget the handle?", answer: "No. The handle retains the provider and resolved root captured at creation or open." }],
    verification: IDENTITY_VERIFICATION,
});
accessor!(
    DATASET_PATH_CATALOG_ENTRY,
    DATASET_PATH_DESCRIPTOR,
    infer_path,
    "Dataset.path",
    "path = Dataset.path(ds)",
    &[d::DATASET],
    &[d::OUT_STRING],
    DatasetForm::Path,
    PATH_DOC,
    LOCAL
);

const ID_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "Dataset.id",
    summary: "Read a Dataset handle's dataset identifier.",
    description: "`id = Dataset.id(ds)` returns the identifier presented when the handle was created or opened; it does not reread the manifest.",
    keywords: &["Dataset.id", "dataset", "identifier"],
    related: &["Dataset.path", "Dataset.version"],
    sections: &[BuiltinDocumentationSection { heading: "Captured metadata", paragraphs: &["The identifier is a text scalar from the handle's captured manifest or provider descriptor. Use `Dataset.refresh` to read a newer descriptor from the same provider and root."] }],
    examples: &[docs::example("dataset-id", "Read a dataset identifier", "schema = struct(); schema.arrays = struct();\nds = data.create('id.data', schema);\nid = Dataset.id(ds);", "assert(~isempty(id));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Is the ID a filesystem path?", answer: "No. Use `Dataset.path` for the public path and `Dataset.id` for the manifest identity." }],
    verification: IDENTITY_VERIFICATION,
});
accessor!(
    DATASET_ID_CATALOG_ENTRY,
    DATASET_ID_DESCRIPTOR,
    infer_id,
    "Dataset.id",
    "id = Dataset.id(ds)",
    &[d::DATASET],
    &[d::OUT_STRING],
    DatasetForm::Id,
    ID_DOC,
    LOCAL
);

const VERSION_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "Dataset.version",
    summary: "Read the version token captured by a Dataset handle.",
    description: "`version = Dataset.version(ds)` returns the handle's presented manifest version token, not a live reread after later writes.",
    keywords: &["Dataset.version", "dataset", "manifest", "optimistic concurrency"],
    related: &["Dataset.refresh", "DataTransaction.commit"],
    sections: &[BuiltinDocumentationSection { heading: "Concurrency", paragraphs: &["A metadata or array write advances the stored manifest sequence. Refresh the handle to obtain a new token before using an `if_manifest` transaction precondition."] }],
    examples: &[docs::example("dataset-version", "Read a captured version", "schema = struct(); schema.arrays = struct();\nds = data.create('version.data', schema);\nv = Dataset.version(ds);", "assert(~isempty(v));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Does the token change inside the same handle after a write?", answer: "No. Call `Dataset.refresh` to obtain a handle with a new presented token." }],
    verification: IDENTITY_VERIFICATION,
});
accessor!(
    DATASET_VERSION_CATALOG_ENTRY,
    DATASET_VERSION_DESCRIPTOR,
    infer_version,
    "Dataset.version",
    "version = Dataset.version(ds)",
    &[d::DATASET],
    &[d::OUT_STRING],
    DatasetForm::Version,
    VERSION_DOC,
    LOCAL
);

const ARRAYS_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "Dataset.arrays",
    summary: "List array names recorded in a dataset manifest.",
    description: "`C = Dataset.arrays(ds)` reads the manifest through the handle's pinned provider and returns a row cell array of array-name strings.",
    keywords: &["Dataset.arrays", "dataset", "manifest", "array names"],
    related: &["Dataset.has_array", "Dataset.array"],
    sections: &[BuiltinDocumentationSection { heading: "Stored names", paragraphs: &["The list is derived from the current manifest and does not load array payloads. An empty dataset returns an empty row cell array."] }],
    examples: &[docs::example("dataset-arrays", "List one array", "schema = struct(); schema.arrays.samples = struct('dtype', 'f64', 'shape', [1 1]);\nds = data.create('arrays.data', schema);\nC = Dataset.arrays(ds);", "assert(numel(C) == 1); assert(strcmp(C{1}, 'samples'));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Are payloads read by this call?", answer: "No. It reads only the manifest's array-name map." }],
    verification: IDENTITY_VERIFICATION,
});
accessor!(
    DATASET_ARRAYS_CATALOG_ENTRY,
    DATASET_ARRAYS_DESCRIPTOR,
    infer_arrays,
    "Dataset.arrays",
    "C = Dataset.arrays(ds)",
    &[d::DATASET],
    &[d::OUT_CELL],
    DatasetForm::Arrays,
    ARRAYS_DOC,
    READ
);

const HAS_ARRAY_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "Dataset.has_array",
    summary: "Check whether a named array is recorded in a dataset.",
    description: "`tf = Dataset.has_array(ds, name)` reads the current manifest through the handle's pinned filesystem owner and returns a logical scalar.",
    keywords: &["Dataset.has_array", "dataset", "array", "membership"],
    related: &["Dataset.arrays", "Dataset.array"],
    sections: &[BuiltinDocumentationSection { heading: "Membership", paragraphs: &["This checks the manifest name map without decoding chunks. The name must be scalar text; a missing name returns false."] }],
    examples: &[docs::example("dataset-has-array", "Check an existing array", "schema = struct(); schema.arrays.samples = struct('dtype', 'f64', 'shape', [1 1]);\nds = data.create('has.data', schema);\ntf = Dataset.has_array(ds, 'samples');", "assert(tf); assert(~Dataset.has_array(ds, 'other'));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Does a true result prove every chunk is readable?", answer: "No. It proves only that the name occurs in the manifest." }],
    verification: IDENTITY_VERIFICATION,
});
accessor!(
    DATASET_HAS_ARRAY_CATALOG_ENTRY,
    DATASET_HAS_ARRAY_DESCRIPTOR,
    infer_has_array,
    "Dataset.has_array",
    "tf = Dataset.has_array(ds, name)",
    &[d::DATASET, d::NAME],
    &[d::OUT_BOOL],
    DatasetForm::HasArray,
    HAS_ARRAY_DOC,
    READ
);

const ARRAY_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "Dataset.array",
    summary: "Open a named DataArray child handle.",
    description: "`arr = Dataset.array(ds, name)` checks the manifest and returns a DataArray handle inheriting the Dataset's exact provider and resolved root.",
    keywords: &["Dataset.array", "DataArray", "dataset", "handle"],
    related: &["Dataset.has_array", "DataArray.read"],
    sections: &[BuiltinDocumentationSection { heading: "Child authority", paragraphs: &["A missing array is an error. The returned child is a managed handle: its displayed dataset path and name are not authorization to reopen the array through a different ambient provider."] }],
    examples: &[docs::example("dataset-array", "Open an array child", "schema = struct(); schema.arrays.samples = struct('dtype', 'f64', 'shape', [1 1]);\nds = data.create('child.data', schema);\narr = Dataset.array(ds, 'samples');", "assert(strcmp(DataArray.name(arr), 'samples'));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Does opening the child load its numeric values?", answer: "No. Call `DataArray.read` to load all values or a selected slice." }],
    verification: IDENTITY_VERIFICATION,
});
accessor!(
    DATASET_ARRAY_CATALOG_ENTRY,
    DATASET_ARRAY_DESCRIPTOR,
    infer_array,
    "Dataset.array",
    "arr = Dataset.array(ds, name)",
    &[d::DATASET, d::NAME],
    &[d::OUT_ARRAY],
    DatasetForm::Array,
    ARRAY_DOC,
    READ
);
