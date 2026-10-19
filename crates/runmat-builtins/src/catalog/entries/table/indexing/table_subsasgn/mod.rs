mod documentation;

pub(super) const ENTRY: crate::BuiltinCatalogEntry = super::entry(
    crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    "table.subsasgn",
    documentation::DOCUMENTATION,
    &super::TABLE_SUBSASGN_DESCRIPTOR,
    super::inference::table_subsasgn,
    true,
    &super::TABLE_SUBSASGN_INTEGER_CAPABILITIES,
);
