mod documentation;
pub(super) const ENTRY: crate::BuiltinCatalogEntry = super::entry(
    crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    "struct2table",
    documentation::DOCUMENTATION,
    &super::STRUCT2TABLE_DESCRIPTOR,
    super::inference::struct2table,
    &[],
    &super::STRUCT2TABLE_INTEGER_CAPABILITIES,
);
