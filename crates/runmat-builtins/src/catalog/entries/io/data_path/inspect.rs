use super::{
    contract::*,
    documentation as docs,
    inference::{self, PathForm},
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    inference::infer(request, entry, PathForm::Inspect)
}

const DOCUMENTATION: BuiltinDocumentation = docs::documentation(docs::PathDocumentationContent {
    name: "data.inspect",
    summary: "Read dataset identity and schema-count metadata.",
    description: "`S = data.inspect(path)` reads the manifest and returns a scalar structure with `path`, `id`, `arrayCount`, and `updatedAt`.",
    keywords: &["data.inspect", "dataset", "metadata", "manifest"],
    related: &["data.open", "data.list", "Dataset.arrays"],
    sections: &[BuiltinDocumentationSection { heading: "Fields and storage", paragraphs: &["`path` is the supplied public path text, `id` is the manifest dataset ID, `arrayCount` is a double scalar count, and `updatedAt` is the manifest timestamp. The captured provider reads and validates the manifest but does not decode array chunks or launch a GPU kernel."] }],
    examples: &[docs::example("inspect-created", "Inspect a new dataset", "schema = struct(); schema.arrays = struct();\ndata.create('work.data', schema);\ninfo = data.inspect('work.data');", "assert(strcmp(info.path, 'work.data')); assert(info.arrayCount == 0); assert(~isempty(info.id));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Does inspect load array values?", answer: "No. It reports manifest metadata only." }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Dataset manifest round trip", location: "builtins::io::data::tests::create_open_write_read_dataset" }],
});

pub const DATA_INSPECT_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "data.inspect",
    DOCUMENTATION,
    &super::descriptor::DATA_INSPECT_DESCRIPTOR,
    infer,
    PathEntryBehavior {
        effects: READ_EFFECTS,
        purity: BuiltinPurity::DeterministicReadOnly,
        integer_capabilities: &[],
        integer_audit: Some(&DATA_NAMESPACE_LIFECYCLE_INTEGER_AUDIT),
    },
);
