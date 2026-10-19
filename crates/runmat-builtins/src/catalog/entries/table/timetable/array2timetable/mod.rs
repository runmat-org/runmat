mod documentation;
pub(super) const ENTRY: crate::BuiltinCatalogEntry =
    super::declarations::entry(super::declarations::EntryDefinition {
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "array2timetable",
        category: "table",
        documentation: documentation::DOCUMENTATION,
        descriptor: &super::ARRAY2TIMETABLE_DESCRIPTOR,
        inference: super::inference::array2timetable,
        extensions: &super::ARRAY2TIMETABLE_EXTENSIONS,
        integer_capabilities: &super::ARRAY2TIMETABLE_INTEGER_CAPABILITIES,
        filesystem_read: false,
    });
