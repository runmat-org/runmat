use super::{
    contract::*,
    documentation as docs,
    inference::{self, PathForm},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    inference::infer(request, entry, PathForm::Copy)
}

const DOCUMENTATION: BuiltinDocumentation = docs::documentation(docs::PathDocumentationContent {
    name: "data.copy",
    summary: "Copy a dataset directory to a new path.",
    description: "`tf = data.copy(fromPath, toPath)` recursively copies a dataset directory and returns true after completion.",
    keywords: &["data.copy", "dataset", "filesystem", "copy"],
    related: &["data.move", "data.import", "data.export"],
    sections: &[BuiltinDocumentationSection { heading: "Stored representation", paragraphs: &["The call copies manifest, indexes, and chunk payload bytes through one captured filesystem provider. It does not decode and recast numeric arrays, so declared integer dtypes and wide integer payload bits are preserved. Source and destination relative paths share the same captured current directory. The copy is not an atomic transaction across the entire directory."] }],
    examples: &[docs::example("copy-created", "Copy a dataset", "schema = struct(); schema.arrays = struct();\ndata.create('source.data', schema);\nok = data.copy('source.data', 'copy.data');\nfound = data.exists('copy.data');", "assert(ok); assert(found);" )],
    faqs: &[BuiltinDocumentationFaq { question: "Does copy convert stored integer arrays?", answer: "No. Dataset files are copied as stored bytes; each array's declared dtype remains in the copied manifest." }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Namespace argument and integer-form boundary", location: "builtins::io::data::tests::data_namespace_lifecycle_rejects_integer_arguments_and_unsupported_extras" }],
});

pub const DATA_COPY_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "data.copy",
    DOCUMENTATION,
    &super::descriptor::DATA_COPY_DESCRIPTOR,
    infer,
    PathEntryBehavior {
        effects: COPY_EFFECTS,
        purity: BuiltinPurity::Impure,
        integer_capabilities: &[],
        integer_audit: Some(&DATA_NAMESPACE_LIFECYCLE_INTEGER_AUDIT),
    },
);
