mod documentation;
pub(super) const ENTRY: crate::BuiltinCatalogEntry = super::entry(
    crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    "timetable2table",
    "table",
    documentation::DOCUMENTATION,
    &super::TIMETABLE2TABLE_DESCRIPTOR,
    super::inference::timetable2table,
    &super::TIMETABLE2TABLE_EXTENSIONS,
    &super::TIMETABLE2TABLE_INTEGER_CAPABILITIES,
    false,
);
