mod documentation;
pub(super) const ENTRY: crate::BuiltinCatalogEntry = super::entry(
    crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    "table2array",
    documentation::DOCUMENTATION,
    &super::TABLE2ARRAY_DESCRIPTOR,
    super::inference::table2array,
    &[],
    &super::TABLE2ARRAY_INTEGER_CAPABILITIES,
);
