use super::{
    contract::*,
    documentation as docs,
    inference::{self, PathForm},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    inference::infer(request, entry, PathForm::Move)
}

const DOCUMENTATION: BuiltinDocumentation = docs::documentation(docs::PathDocumentationContent {
    name: "data.move",
    summary: "Move a dataset directory to a new path.",
    description: "`tf = data.move(fromPath, toPath)` asks the filesystem provider to rename the dataset path and returns true on success.",
    keywords: &["data.move", "dataset", "filesystem", "rename"],
    related: &["data.copy", "data.delete", "data.open"],
    sections: &[BuiltinDocumentationSection { heading: "Provider rename", paragraphs: &["Both paths are resolved with the same captured filesystem provider and current directory. The provider's rename semantics and errors apply; RunMat does not emulate a failed rename by copying or replaying against a replacement provider. Existing Dataset handles retain their original provider and resolved root; moving the directory does not retarget them to the new path."] }],
    examples: &[docs::example("move-created", "Rename a dataset", "schema = struct(); schema.arrays = struct();\ndata.create('before.data', schema);\nok = data.move('before.data', 'after.data');\nfound = data.exists('after.data');", "assert(ok); assert(found); assert(~data.exists('before.data'));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Does moving update an existing Dataset handle?", answer: "No. Reopen the new path to obtain a handle whose public path refers to the moved directory." }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Namespace argument and integer-form boundary", location: "builtins::io::data::tests::data_namespace_lifecycle_rejects_integer_arguments_and_unsupported_extras" }],
});

pub const DATA_MOVE_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "data.move",
    DOCUMENTATION,
    &super::descriptor::DATA_MOVE_DESCRIPTOR,
    infer,
    PathEntryBehavior {
        effects: COPY_EFFECTS,
        purity: BuiltinPurity::Impure,
        integer_capabilities: &[],
        integer_audit: Some(&DATA_NAMESPACE_LIFECYCLE_INTEGER_AUDIT),
    },
);
