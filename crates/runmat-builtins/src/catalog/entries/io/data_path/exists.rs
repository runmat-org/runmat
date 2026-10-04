use super::{
    contract::*,
    documentation as docs,
    inference::{self, PathForm},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    inference::infer(request, entry, PathForm::Exists)
}

const DOCUMENTATION: BuiltinDocumentation = docs::documentation(docs::PathDocumentationContent {
    name: "data.exists",
    summary: "Check whether a dataset manifest exists.",
    description: "`tf = data.exists(path)` returns a logical scalar indicating whether the resolved path has a readable manifest.",
    keywords: &["data.exists", "dataset", "filesystem"],
    related: &["data.open", "data.list", "data.inspect"],
    sections: &[BuiltinDocumentationSection { heading: "Meaning of existence", paragraphs: &["This checks metadata for `manifest.json` through the captured filesystem provider. A missing or unreadable manifest returns false; it does not parse the manifest or validate chunk payloads. The call is host filesystem I/O and never launches a GPU kernel."] }],
    examples: &[docs::example("created-exists", "Check an existing dataset", "schema = struct(); schema.arrays = struct();\ndata.create('work.data', schema);\nfound = data.exists('work.data');\nmissing = data.exists('absent.data');", "assert(found); assert(~missing);" )],
    faqs: &[BuiltinDocumentationFaq { question: "Does true prove the dataset is readable?", answer: "No. It establishes that manifest metadata is accessible; data.open parses and validates the manifest." }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Dataset create and existence", location: "builtins::io::data::tests::create_open_write_read_dataset" }],
});

pub const DATA_EXISTS_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "data.exists",
    DOCUMENTATION,
    &super::descriptor::DATA_EXISTS_DESCRIPTOR,
    infer,
    PathEntryBehavior {
        effects: READ_EFFECTS,
        purity: BuiltinPurity::DeterministicReadOnly,
        integer_capabilities: &[],
        integer_audit: Some(&DATA_NAMESPACE_LIFECYCLE_INTEGER_AUDIT),
    },
);
