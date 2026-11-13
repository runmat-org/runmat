use super::{
    contract::*,
    documentation as docs,
    inference::{self, PathForm},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    inference::infer(request, entry, PathForm::List)
}

const DOCUMENTATION: BuiltinDocumentation = docs::documentation(docs::PathDocumentationContent {
    name: "data.list",
    summary: "List child dataset directories under a prefix.",
    description: "`C = data.list(prefix)` returns a one-row cell array of qualifying child `.data` paths.",
    keywords: &["data.list", "dataset", "directory", "discovery"],
    related: &["data.exists", "data.inspect", "data.open"],
    sections: &[BuiltinDocumentationSection { heading: "Enumeration", paragraphs: &["The captured filesystem provider enumerates the resolved prefix. Entries must be directories with a `.data` extension and accessible `manifest.json` metadata. Returned strings preserve the supplied prefix text joined to each child name; manifests are not parsed. A directory-read error is returned; an unreadable child manifest is omitted."] }],
    examples: &[docs::example("list-created", "List datasets under the current directory", "schema = struct(); schema.arrays = struct();\ndata.create('listed.data', schema);\npaths = data.list('.');", "assert(numel(paths) == 1); assert(strcmp(paths{1}, './listed.data'));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Does list search recursively?", answer: "No. It examines immediate child directories of the supplied prefix." }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Namespace argument and integer-form boundary", location: "builtins::io::data::tests::data_namespace_lifecycle_rejects_integer_arguments_and_unsupported_extras" }],
});

pub const DATA_LIST_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "data.list",
    DOCUMENTATION,
    &super::descriptor::DATA_LIST_DESCRIPTOR,
    infer,
    PathEntryBehavior {
        effects: READ_EFFECTS,
        purity: BuiltinPurity::DeterministicReadOnly,
        integer_capabilities: &[],
        integer_audit: Some(&DATA_NAMESPACE_LIFECYCLE_INTEGER_AUDIT),
    },
);
