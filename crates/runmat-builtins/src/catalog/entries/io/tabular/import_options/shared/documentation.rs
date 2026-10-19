use crate::*;

pub(crate) const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Typed import options and selectors",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/tabular/import_options/model.rs",
    ),
};
pub(crate) const TABLE_TEST: BuiltinEvidenceReference = BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Per-variable table import, missing, separators, and datetime format",
    location: "builtins::table::tests::variable_import_options_modify_real_table_columns_and_preserve_object_identity",
};
