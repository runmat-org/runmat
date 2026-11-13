use super::{
    contract::*,
    documentation as docs,
    inference::{self, PathForm},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    inference::infer(request, entry, PathForm::Import)
}

const DOCUMENTATION: BuiltinDocumentation = docs::documentation(docs::PathDocumentationContent {
    name: "data.import",
    summary: "Import a RunMat dataset directory.",
    description: "`ds = data.import(path, 'data', sourcePath)` copies a stored dataset into `path`, validates the copied manifest, and returns a Dataset handle.",
    keywords: &["data.import", "dataset", "filesystem", "import"],
    related: &["data.export", "data.copy", "data.open"],
    sections: &[BuiltinDocumentationSection { heading: "Format and effects", paragraphs: &["Only the case-insensitive `data` format token is currently supported. Unsupported formats fail before filesystem copying. Source and destination use the same captured provider and current directory; stored array payload bytes retain their declared numeric classes. If manifest validation fails after copying, the provider error is returned and the copied directory is not automatically rolled back."] }],
    examples: &[docs::example("import-created", "Import a dataset directory", "schema = struct(); schema.arrays = struct();\ndata.create('source.data', schema);\nds = data.import('imported.data', 'data', 'source.data');\nfound = data.exists('imported.data');", "assert(found); assert(strcmp(Dataset.path(ds), 'imported.data'));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Can I import an arbitrary MAT or CSV file?", answer: "No. This form currently accepts another RunMat `.data` directory only." }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Namespace format and argument boundary", location: "builtins::io::data::tests::data_namespace_lifecycle_rejects_integer_arguments_and_unsupported_extras" }],
});

pub const DATA_IMPORT_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "data.import",
    DOCUMENTATION,
    &super::descriptor::DATA_IMPORT_DESCRIPTOR,
    infer,
    PathEntryBehavior {
        effects: COPY_EFFECTS,
        purity: BuiltinPurity::Impure,
        integer_capabilities: &[],
        integer_audit: Some(&DATA_NAMESPACE_LIFECYCLE_INTEGER_AUDIT),
    },
);
