mod documentation;
pub(super) const ENTRY: crate::BuiltinCatalogEntry =
    super::declarations::entry(super::declarations::EntryDefinition {
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "table2timetable",
        category: "table",
        documentation: documentation::DOCUMENTATION,
        descriptor: &super::TABLE2TIMETABLE_DESCRIPTOR,
        inference: super::inference::table2timetable,
        extensions: &super::TABLE2TIMETABLE_EXTENSIONS,
        integer_capabilities: &super::TABLE2TIMETABLE_INTEGER_CAPABILITIES,
        filesystem_read: false,
    });
