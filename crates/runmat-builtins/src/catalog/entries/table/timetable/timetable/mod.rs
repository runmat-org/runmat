mod documentation;
pub(super) const ENTRY: crate::BuiltinCatalogEntry = super::entry(
    crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    "timetable",
    "table",
    documentation::DOCUMENTATION,
    &super::TIMETABLE_DESCRIPTOR,
    super::inference::timetable,
    &super::TIMETABLE_EXTENSIONS,
    &super::TIMETABLE_INTEGER_CAPABILITIES,
    false,
);
