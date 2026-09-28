mod documentation;
pub(super) const ENTRY: crate::BuiltinCatalogEntry = super::entry(
    crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    "array2timetable",
    "table",
    documentation::DOCUMENTATION,
    &super::ARRAY2TIMETABLE_DESCRIPTOR,
    super::inference::array2timetable,
    &super::ARRAY2TIMETABLE_EXTENSIONS,
    &super::ARRAY2TIMETABLE_INTEGER_CAPABILITIES,
    false,
);
