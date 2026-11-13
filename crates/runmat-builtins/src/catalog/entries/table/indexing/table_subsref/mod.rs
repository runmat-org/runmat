mod documentation;

pub(super) const ENTRY: crate::BuiltinCatalogEntry = super::entry(
    crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    "table.subsref",
    documentation::DOCUMENTATION,
    &super::TABLE_SUBSREF_DESCRIPTOR,
    super::inference::table_subsref,
    false,
    &super::TABLE_SUBSREF_INTEGER_CAPABILITIES,
);
