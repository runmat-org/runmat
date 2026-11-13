mod documentation;
pub(super) const ENTRY: crate::BuiltinCatalogEntry = super::entry(
    crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    "table2struct",
    documentation::DOCUMENTATION,
    &super::TABLE2STRUCT_DESCRIPTOR,
    super::inference::table2struct,
    &super::TABLE2STRUCT_EXTENSIONS,
    &super::TABLE2STRUCT_INTEGER_CAPABILITIES,
);
