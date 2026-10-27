//! Dataset transaction creation, manifest snapshot, and refresh contracts.

use super::super::{
    contract::{self, Behavior, BEGIN_EFFECTS, FILESYSTEM, READ_EFFECTS, SNAPSHOT_EFFECTS},
    descriptor as d, documentation as docs,
    inference::{self, DatasetForm},
    integer::DATASET_NONATTRIBUTE_AUDIT,
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

macro_rules! lifecycle {
    ($entry:ident, $descriptor:ident, $infer:ident, $name:literal, $label:literal, $inputs:expr, $output:expr, $form:expr, $doc:expr, $effects:expr, $purity:expr) => {
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
            Behavior {
                effects: $effects,
                capabilities: FILESYSTEM,
                purity: $purity,
                async_behavior: BuiltinAsyncBehavior::MaySuspend,
                integer_capabilities: &[],
                integer_audit: Some(&DATASET_NONATTRIBUTE_AUDIT),
            },
        );
    };
}

const LIFECYCLE_VERIFICATION: &[BuiltinEvidenceReference] = &[BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Dataset owner and transaction round trip",
    location: "builtins::io::data::tests::data_create_retains_selected_provider_and_relative_cwd_after_first_write",
}];

const BEGIN_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "Dataset.begin",
    summary: "Begin a process-local Dataset transaction.",
    description: "`tx = Dataset.begin(ds)` reads the manifest sequence and returns a managed DataTransaction child pinned to the Dataset's provider and resolved root.",
    keywords: &["Dataset.begin", "DataTransaction", "dataset", "transaction"],
    related: &["DataTransaction.commit", "DataTransaction.abort"],
    sections: &[BuiltinDocumentationSection { heading: "Concurrency", paragraphs: &["Queued writes remain process-local until commit. Commit checks the manifest sequence, writes array changes into fresh revisions, and publishes their references through the manifest. Rejected publication preserves previously referenced array objects. Failed writes can leave unreferenced revision files; commit is not a filesystem-wide transaction."] }],
    examples: &[docs::example("dataset-begin", "Begin and abort a transaction", "schema = struct(); schema.arrays = struct();\nds = data.create('begin.data', schema);\ntx = Dataset.begin(ds);\nstate = DataTransaction.status(tx);", "assert(strcmp(state, 'open')); assert(DataTransaction.abort(tx));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Can a transaction move to another provider?", answer: "No. It retains the Dataset handle's exact provider and root." }],
    verification: LIFECYCLE_VERIFICATION,
});
lifecycle!(
    DATASET_BEGIN_CATALOG_ENTRY,
    DATASET_BEGIN_DESCRIPTOR,
    infer_begin,
    "Dataset.begin",
    "tx = Dataset.begin(ds)",
    &[d::DATASET],
    &[d::OUT_TRANSACTION],
    DatasetForm::Begin,
    BEGIN_DOC,
    BEGIN_EFFECTS,
    BuiltinPurity::Impure
);

const SNAPSHOT_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "Dataset.snapshot",
    summary: "Copy the current manifest into the dataset's snapshot directory.",
    description: "`snapshotPath = Dataset.snapshot(ds, label)` copies `manifest.json` to `.snapshots/<sanitized-label>.manifest.json` through the handle's pinned provider.",
    keywords: &["Dataset.snapshot", "dataset", "manifest", "snapshot"],
    related: &["Dataset.version", "Dataset.refresh"],
    sections: &[BuiltinDocumentationSection { heading: "Scope", paragraphs: &["Only the manifest file is copied; array payloads and chunk indexes are not snapshotted. Non-ASCII-alphanumeric label characters other than hyphen and underscore become underscores, so different labels can target the same filename."] }],
    examples: &[docs::example("dataset-snapshot", "Snapshot a manifest", "schema = struct(); schema.arrays = struct();\nds = data.create('snapshot.data', schema);\np = Dataset.snapshot(ds, 'baseline');", "assert(~isempty(fileread(p)));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Is this a complete immutable dataset backup?", answer: "No. It copies the manifest only and can overwrite a same-label snapshot." }],
    verification: LIFECYCLE_VERIFICATION,
});
lifecycle!(
    DATASET_SNAPSHOT_CATALOG_ENTRY,
    DATASET_SNAPSHOT_DESCRIPTOR,
    infer_snapshot,
    "Dataset.snapshot",
    "snapshotPath = Dataset.snapshot(ds, label)",
    &[d::DATASET, d::LABEL],
    &[d::OUT_STRING],
    DatasetForm::Snapshot,
    SNAPSHOT_DOC,
    SNAPSHOT_EFFECTS,
    BuiltinPurity::Impure
);

const REFRESH_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "Dataset.refresh",
    summary: "Read the latest manifest into a new Dataset handle.",
    description: "`fresh = Dataset.refresh(ds)` rereads the manifest through the original handle's provider and resolved root, then returns a new managed handle with updated ID/version presentation.",
    keywords: &["Dataset.refresh", "dataset", "manifest", "version"],
    related: &["Dataset.version", "data.open"],
    sections: &[BuiltinDocumentationSection { heading: "Owner retention", paragraphs: &["Refreshing does not switch to the current ambient provider or working directory. The original handle remains valid and keeps its previously captured version token."] }],
    examples: &[docs::example("dataset-refresh", "Refresh a version token", "schema = struct(); schema.arrays = struct();\nds = data.create('refresh.data', schema);\nDataset.set_attr(ds, 'stage', 'ready');\nfresh = Dataset.refresh(ds);", "assert(~strcmp(Dataset.version(ds), Dataset.version(fresh)));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Does refresh decode numeric arrays?", answer: "No. It rereads the manifest and provider descriptor, not array chunks." }],
    verification: LIFECYCLE_VERIFICATION,
});
lifecycle!(
    DATASET_REFRESH_CATALOG_ENTRY,
    DATASET_REFRESH_DESCRIPTOR,
    infer_refresh,
    "Dataset.refresh",
    "fresh = Dataset.refresh(ds)",
    &[d::DATASET],
    &[d::OUT_DATASET],
    DatasetForm::Refresh,
    REFRESH_DOC,
    READ_EFFECTS,
    BuiltinPurity::DeterministicReadOnly
);
