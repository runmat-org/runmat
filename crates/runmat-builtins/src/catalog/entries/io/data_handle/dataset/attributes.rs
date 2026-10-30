//! Dataset manifest attribute read and mutation contracts.

use super::super::{
    contract::{self, Behavior, FILESYSTEM, READ_EFFECTS, WRITE_EFFECTS},
    descriptor as d, documentation as docs,
    inference::{self, DatasetForm},
    integer,
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

macro_rules! attribute {
    ($entry:ident, $descriptor:ident, $infer:ident, $name:literal, $label:literal, $inputs:expr, $output:expr, $form:expr, $doc:expr, $effects:expr, $purity:expr, $integer:expr) => {
        d::receiver_descriptor!($descriptor, $label, $inputs, $output);
        fn $infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
            inference::dataset_infer(request, entry, $form)
        }
        pub const $entry: BuiltinCatalogEntry = contract::entry(
            BuiltinCatalogProvenance::new(file!(), module_path!()),
            $name,
            $doc,
            &$descriptor,
            $infer,
            Behavior {
                effects: $effects,
                capabilities: FILESYSTEM,
                purity: $purity,
                async_behavior: BuiltinAsyncBehavior::MaySuspend,
                integer_capabilities: $integer,
                integer_audit: None,
            },
        );
    };
}

const ATTRIBUTE_VERIFICATION: &[BuiltinEvidenceReference] = &[BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Exact wide integer manifest attributes",
    location: "builtins::io::data::tests::dataset_attributes_keep_exact_integer_values_and_documented_class_boundaries",
}, BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Scalar JSON admission and typed string presence",
    location: "builtins::io::data::tests::attribute_json_roundtrips_exact_scalars_and_rejects_untyped_aggregates",
}];

const ATTRS_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "Dataset.attrs",
    summary: "Read all stored Dataset attributes as a structure.",
    description: "`S = Dataset.attrs(ds)` reads the current manifest through the handle's pinned provider and converts its JSON attributes to a scalar struct.",
    keywords: &["Dataset.attrs", "dataset", "metadata", "attributes"],
    related: &["Dataset.get_attr", "Dataset.set_attrs"],
    sections: &[BuiltinDocumentationSection { heading: "Stored values", paragraphs: &["JSON preserves exact integer mathematical values but not narrow integer storage classes; stored values decode as int64 when representable, otherwise uint64. JSON null decodes as a typed missing string; literal `<missing>` text remains ordinary text."] }],
    examples: &[docs::example("dataset-attrs", "Read attribute fields", "schema = struct(); schema.arrays = struct();\nds = data.create('attrs.data', schema);\nDataset.set_attr(ds, 'source', 'sensor');\nS = Dataset.attrs(ds);", "assert(strcmp(S.source, 'sensor'));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Does this load array chunks?", answer: "No. It reads only the manifest attribute map." }],
    verification: ATTRIBUTE_VERIFICATION,
});
attribute!(
    DATASET_ATTRS_CATALOG_ENTRY,
    DATASET_ATTRS_DESCRIPTOR,
    infer_attrs,
    "Dataset.attrs",
    "S = Dataset.attrs(ds)",
    &[d::DATASET],
    &[d::OUT_STRUCT],
    DatasetForm::Attrs,
    ATTRS_DOC,
    READ_EFFECTS,
    BuiltinPurity::DeterministicReadOnly,
    integer::DATASET_ATTRS
);

const GET_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "Dataset.get_attr",
    summary: "Read one Dataset attribute or return a fallback.",
    description: "`value = Dataset.get_attr(ds, key, defaultValue)` reads the manifest and returns a stored attribute; a missing key returns the optional default directly, or double zero when omitted.",
    keywords: &["Dataset.get_attr", "dataset", "attribute", "default"],
    related: &["Dataset.attrs", "Dataset.set_attr"],
    sections: &[BuiltinDocumentationSection { heading: "Class behavior", paragraphs: &["A stored JSON integer decodes exactly as int64 or uint64 according to magnitude. Stored JSON null returns a typed missing string, not the default. Literal `<missing>` text remains ordinary text. An explicitly supplied default bypasses JSON and retains its original class and value."] }],
    examples: &[docs::example("dataset-get-attr", "Read an attribute with fallback", "schema = struct(); schema.arrays = struct();\nds = data.create('getattr.data', schema);\nv = Dataset.get_attr(ds, 'missing', 17);", "assert(v == 17);" )],
    faqs: &[BuiltinDocumentationFaq { question: "What happens without a default?", answer: "A missing key returns double zero." }],
    verification: ATTRIBUTE_VERIFICATION,
});
attribute!(
    DATASET_GET_ATTR_CATALOG_ENTRY,
    DATASET_GET_ATTR_DESCRIPTOR,
    infer_get_attr,
    "Dataset.get_attr",
    "value = Dataset.get_attr(ds, key, defaultValue)",
    &[d::DATASET, d::KEY, d::DEFAULT],
    &[d::OUT_VALUE],
    DatasetForm::GetAttr,
    GET_DOC,
    READ_EFFECTS,
    BuiltinPurity::DeterministicReadOnly,
    integer::DATASET_GET_ATTR
);

const SET_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "Dataset.set_attr",
    summary: "Set one Dataset manifest attribute.",
    description: "`tf = Dataset.set_attr(ds, key, value)` updates one manifest attribute, timestamp, and sequence through the handle's pinned filesystem provider.",
    keywords: &["Dataset.set_attr", "dataset", "attribute", "metadata"],
    related: &["Dataset.get_attr", "Dataset.set_attrs"],
    sections: &[BuiltinDocumentationSection { heading: "Persistence", paragraphs: &[docs::ATTRIBUTE_VALUE_BOUNDARY, "Scalar integer values, including full-width int64 and uint64, serialize as exact JSON numbers. The operation advances the stored manifest version; the original handle's displayed version remains its captured token until refresh."] }],
    examples: &[docs::example("dataset-set-attr", "Store an attribute", "schema = struct(); schema.arrays = struct();\nds = data.create('setattr.data', schema);\ntf = Dataset.set_attr(ds, 'stage', 'ready');", "assert(tf); assert(strcmp(Dataset.get_attr(ds, 'stage'), 'ready'));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Is the update atomic across all dataset files?", answer: "It writes the manifest; broader multi-file atomicity is not provided by this call." }],
    verification: ATTRIBUTE_VERIFICATION,
});
attribute!(
    DATASET_SET_ATTR_CATALOG_ENTRY,
    DATASET_SET_ATTR_DESCRIPTOR,
    infer_set_attr,
    "Dataset.set_attr",
    "tf = Dataset.set_attr(ds, key, value)",
    &[d::DATASET, d::KEY, d::VALUE],
    &[d::OUT_BOOL],
    DatasetForm::SetAttr,
    SET_DOC,
    WRITE_EFFECTS,
    BuiltinPurity::Impure,
    integer::DATASET_SET_ATTR
);

const SETS_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "Dataset.set_attrs",
    summary: "Set several Dataset manifest attributes.",
    description: "`tf = Dataset.set_attrs(ds, attrs)` merges the fields of a scalar attribute structure into one manifest update.",
    keywords: &["Dataset.set_attrs", "dataset", "attributes", "metadata"],
    related: &["Dataset.attrs", "Dataset.set_attr"],
    sections: &[BuiltinDocumentationSection { heading: "Persistence", paragraphs: &[docs::ATTRIBUTE_VALUE_BOUNDARY, "The update advances the manifest timestamp and sequence once. Scalar integer fields retain their exact mathematical values as JSON numbers, though JSON does not preserve a narrow integer storage class."] }],
    examples: &[docs::example("dataset-set-attrs", "Store two attributes", "schema = struct(); schema.arrays = struct();\nds = data.create('setattrs.data', schema);\na = struct('source', 'sensor', 'stage', 'ready');\ntf = Dataset.set_attrs(ds, a);", "assert(tf); assert(strcmp(Dataset.get_attr(ds, 'source'), 'sensor')); assert(strcmp(Dataset.get_attr(ds, 'stage'), 'ready'));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Does this replace every existing field?", answer: "No. Supplied fields are merged into the existing manifest attribute map." }],
    verification: ATTRIBUTE_VERIFICATION,
});
attribute!(
    DATASET_SET_ATTRS_CATALOG_ENTRY,
    DATASET_SET_ATTRS_DESCRIPTOR,
    infer_set_attrs,
    "Dataset.set_attrs",
    "tf = Dataset.set_attrs(ds, attrs)",
    &[d::DATASET, d::ATTRS],
    &[d::OUT_BOOL],
    DatasetForm::SetAttrs,
    SETS_DOC,
    WRITE_EFFECTS,
    BuiltinPurity::Impure,
    integer::DATASET_SET_ATTRS
);
