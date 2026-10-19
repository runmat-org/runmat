use super::{
    contract::*,
    documentation as docs,
    inference::{self, PathForm},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    inference::infer(request, entry, PathForm::Export)
}

const DOCUMENTATION: BuiltinDocumentation = docs::documentation(docs::PathDocumentationContent {
    name: "data.export",
    summary: "Export a RunMat dataset directory to another path.",
    description: "`tf = data.export(path, 'data', targetPath)` copies the dataset directory and returns true after completion.",
    keywords: &["data.export", "dataset", "filesystem", "export"],
    related: &["data.import", "data.copy", "data.inspect"],
    sections: &[BuiltinDocumentationSection { heading: "Format and representation", paragraphs: &["Only the case-insensitive `data` format token is supported. Unsupported formats fail before filesystem effects. The manifest and chunk payloads are copied as stored bytes through a captured provider, retaining fixed-width integer payloads exactly. The operation does not convert to MAT, CSV, or another file format, and a partial destination is possible on provider failure."] }],
    examples: &[docs::example("export-created", "Export a dataset directory", "schema = struct(); schema.arrays = struct();\ndata.create('source.data', schema);\nok = data.export('source.data', 'data', 'exported.data');\nfound = data.exists('exported.data');", "assert(ok); assert(found);" )],
    faqs: &[BuiltinDocumentationFaq { question: "Is export a format conversion?", answer: "No. At present it copies a RunMat `.data` directory in the same storage format." }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Namespace format and argument boundary", location: "builtins::io::data::tests::data_namespace_lifecycle_rejects_integer_arguments_and_unsupported_extras" }],
});

pub const DATA_EXPORT_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "data.export",
    DOCUMENTATION,
    &super::descriptor::DATA_EXPORT_DESCRIPTOR,
    infer,
    PathEntryBehavior {
        effects: COPY_EFFECTS,
        purity: BuiltinPurity::Impure,
        integer_capabilities: &[],
        integer_audit: Some(&DATA_NAMESPACE_LIFECYCLE_INTEGER_AUDIT),
    },
);
