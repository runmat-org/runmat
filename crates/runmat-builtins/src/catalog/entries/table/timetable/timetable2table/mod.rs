mod documentation;
pub(super) const ENTRY: crate::BuiltinCatalogEntry =
    super::declarations::entry(super::declarations::EntryDefinition {
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "timetable2table",
        category: "table",
        documentation: documentation::DOCUMENTATION,
        descriptor: &super::TIMETABLE2TABLE_DESCRIPTOR,
        inference: super::inference::timetable2table,
        extensions: &super::TIMETABLE2TABLE_EXTENSIONS,
        integer_capabilities: &super::TIMETABLE2TABLE_INTEGER_CAPABILITIES,
        filesystem_read: false,
    });
