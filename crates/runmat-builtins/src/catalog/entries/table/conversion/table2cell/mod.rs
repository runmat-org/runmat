mod documentation;
pub(super) const ENTRY: crate::BuiltinCatalogEntry = super::entry(
    crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    "table2cell",
    documentation::DOCUMENTATION,
    &super::TABLE2CELL_DESCRIPTOR,
    super::inference::table2cell,
    &[],
    &super::TABLE2CELL_INTEGER_CAPABILITIES,
);
