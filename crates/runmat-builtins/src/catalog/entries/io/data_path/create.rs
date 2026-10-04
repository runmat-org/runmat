use super::{
    contract::*,
    documentation as docs,
    inference::{self, PathForm},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    inference::infer(request, entry, PathForm::Create)
}

const DOCUMENTATION: BuiltinDocumentation = docs::documentation(docs::PathDocumentationContent {
    name: "data.create",
    summary: "Create a typed dataset at a .data path.",
    description: "`ds = data.create(path, schema)` writes a new manifest and zero-filled, chunked arrays, returning a Dataset handle.",
    keywords: &["data.create", "dataset", "schema", "chunked storage"],
    related: &["data.open", "Dataset.array", "DataArray.write"],
    sections: &[
        BuiltinDocumentationSection { heading: "Schema and integer controls", paragraphs: &["`schema.arrays` is a structure keyed by array name. Each array may specify `dtype` (default f64), `shape` (default [0 0]), `chunk`, and `codec`. Supported dtypes are f64, f32, and all eight signed/unsigned fixed-width integer classes. Shapes and chunks accept exact typed integers or integral single/double dimensions; invalid, negative, overflowing, or rank-mismatched dimensions are rejected before array publication."] },
        BuiltinDocumentationSection { heading: "Filesystem authority", paragraphs: &["The call uses one captured filesystem provider and current directory for its manifest and all chunk payloads. Relative paths are resolved against that directory. This is host filesystem work, not a GPU kernel; nested resident numeric values are not accepted as schema dimensions without their documented structural conversion."] },
    ],
    examples: &[docs::example("empty-dataset", "Create an empty dataset", "schema = struct();\nschema.arrays = struct();\nds = data.create('empty.data', schema);\nfound = data.exists('empty.data');", "assert(found); assert(strcmp(Dataset.path(ds), 'empty.data'));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Can I create an integer array?", answer: "Yes. Set `dtype` to an integer class such as `uint64`; shape and chunk controls are checked without reducing wide integers through double." }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed schema dimensions and integer payloads", location: "builtins::io::data::tests::data_create_uses_exact_all_class_schema_controls_and_integer_payloads" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Pinned provider and relative directory", location: "builtins::io::data::tests::data_create_retains_selected_provider_and_relative_cwd_after_first_write" }],
});

pub const DATA_CREATE_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "data.create",
    DOCUMENTATION,
    &super::descriptor::DATA_CREATE_DESCRIPTOR,
    infer,
    PathEntryBehavior {
        effects: CREATE_EFFECTS,
        purity: BuiltinPurity::Impure,
        integer_capabilities: super::descriptor::DATA_CREATE_INTEGER_CAPABILITIES,
        integer_audit: None,
    },
);
