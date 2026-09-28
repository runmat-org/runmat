mod documentation;
pub(super) const ENTRY: crate::BuiltinCatalogEntry = super::entry(
    crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    "table2timetable",
    "table",
    documentation::DOCUMENTATION,
    &super::TABLE2TIMETABLE_DESCRIPTOR,
    super::inference::table2timetable,
    &super::TABLE2TIMETABLE_EXTENSIONS,
    &super::TABLE2TIMETABLE_INTEGER_CAPABILITIES,
    false,
);
