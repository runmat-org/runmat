use super::{
    contract::*,
    documentation as docs,
    inference::{self, PathForm},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    inference::infer(request, entry, PathForm::Open)
}

const DOCUMENTATION: BuiltinDocumentation = docs::documentation(docs::PathDocumentationContent {
    name: "data.open",
    summary: "Open a stored dataset by path.",
    description: "`ds = data.open(path)` validates a dataset manifest and returns its Dataset handle and current version facts.",
    keywords: &["data.open", "dataset", "manifest"],
    related: &["data.create", "data.inspect", "Dataset.array"],
    sections: &[BuiltinDocumentationSection { heading: "Handle and storage", paragraphs: &["The manifest is read through the captured filesystem provider and current directory. The returned Dataset object presents the public path, dataset ID, and version token while retaining a private lease for that exact provider and resolved root; later Dataset and DataArray calls use that owner even if the ambient provider or directory changes. Arrays are not decoded by opening the handle."] }],
    examples: &[docs::example("reopen-created", "Reopen a dataset", "schema = struct(); schema.arrays = struct();\ndata.create('work.data', schema);\nds = data.open('work.data');\nid = Dataset.id(ds);", "assert(~isempty(id)); assert(strcmp(Dataset.path(ds), 'work.data'));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Does opening load array payloads?", answer: "No. It validates and reads the manifest; use Dataset.array and DataArray.read to access stored arrays." }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Create/open dataset round trip", location: "builtins::io::data::tests::create_open_write_read_dataset" }],
});

pub const DATA_OPEN_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "data.open",
    DOCUMENTATION,
    &super::descriptor::DATA_OPEN_DESCRIPTOR,
    infer,
    PathEntryBehavior {
        effects: READ_EFFECTS,
        purity: BuiltinPurity::DeterministicReadOnly,
        integer_capabilities: &[],
        integer_audit: Some(&DATA_NAMESPACE_LIFECYCLE_INTEGER_AUDIT),
    },
);
