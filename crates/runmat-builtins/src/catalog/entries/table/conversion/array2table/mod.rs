mod documentation;
pub(super) const ENTRY: crate::BuiltinCatalogEntry = super::entry(
    crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    "array2table",
    documentation::DOCUMENTATION,
    &super::ARRAY2TABLE_DESCRIPTOR,
    super::inference::array2table,
    &super::ARRAY2TABLE_EXTENSIONS,
    &super::ARRAY2TABLE_INTEGER_CAPABILITIES,
);
