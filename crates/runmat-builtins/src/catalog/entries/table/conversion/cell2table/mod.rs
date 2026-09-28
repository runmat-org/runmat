mod documentation;
pub(super) const ENTRY: crate::BuiltinCatalogEntry = super::entry(
    crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    "cell2table",
    documentation::DOCUMENTATION,
    &super::CELL2TABLE_DESCRIPTOR,
    super::inference::cell2table,
    &[],
    &super::CELL2TABLE_INTEGER_CAPABILITIES,
);
