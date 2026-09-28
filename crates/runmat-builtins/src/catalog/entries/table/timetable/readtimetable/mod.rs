mod documentation;
pub(super) const ENTRY: crate::BuiltinCatalogEntry = super::entry(
    crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    "readtimetable",
    "io/tabular",
    documentation::DOCUMENTATION,
    &super::READTIMETABLE_DESCRIPTOR,
    super::inference::readtimetable,
    &super::READTIMETABLE_EXTENSIONS,
    &super::READTIMETABLE_INTEGER_CAPABILITIES,
    true,
);
