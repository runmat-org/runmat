use super::{
    contract::*,
    documentation as docs,
    inference::{self, PathForm},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    inference::infer(request, entry, PathForm::Delete)
}

const DOCUMENTATION: BuiltinDocumentation = docs::documentation(docs::PathDocumentationContent {
    name: "data.delete",
    summary: "Remove a dataset directory.",
    description: "`tf = data.delete(path)` removes the resolved dataset path recursively and returns true when the provider completes the removal.",
    keywords: &["data.delete", "dataset", "filesystem", "remove"],
    related: &["data.create", "data.exists", "data.move"],
    sections: &[BuiltinDocumentationSection { heading: "Destructive effect", paragraphs: &["This removes the entire directory, including manifest, chunk indexes, and payloads, through the captured filesystem provider. The operation is not transactional and a provider error is returned without replay on another provider. Confirm the path before calling."] }],
    examples: &[docs::example("delete-created", "Remove a scratch dataset", "schema = struct(); schema.arrays = struct();\ndata.create('scratch.data', schema);\nremoved = data.delete('scratch.data');\nfound = data.exists('scratch.data');", "assert(removed); assert(~found);" )],
    faqs: &[BuiltinDocumentationFaq { question: "Does this remove just the manifest?", answer: "No. The complete dataset directory is removed recursively." }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Dataset lifecycle", location: "builtins::io::data::tests::create_open_write_read_dataset" }],
});

pub const DATA_DELETE_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "data.delete",
    DOCUMENTATION,
    &super::descriptor::DATA_DELETE_DESCRIPTOR,
    infer,
    PathEntryBehavior {
        effects: WRITE_EFFECTS,
        purity: BuiltinPurity::Impure,
        integer_capabilities: &[],
        integer_audit: Some(&DATA_NAMESPACE_LIFECYCLE_INTEGER_AUDIT),
    },
);
