mod documentation;
pub(super) const ENTRY: crate::BuiltinCatalogEntry =
    super::declarations::entry(super::declarations::EntryDefinition {
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "timetable",
        category: "table",
        documentation: documentation::DOCUMENTATION,
        descriptor: &super::TIMETABLE_DESCRIPTOR,
        inference: super::inference::timetable,
        extensions: &super::TIMETABLE_EXTENSIONS,
        integer_capabilities: &super::TIMETABLE_INTEGER_CAPABILITIES,
        filesystem_read: false,
    });
