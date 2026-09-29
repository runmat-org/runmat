mod documentation;
pub(super) const ENTRY: crate::BuiltinCatalogEntry = super::entry(super::EntryDefinition {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "readtimetable",
    category: "io/tabular",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::READTIMETABLE_DESCRIPTOR,
    inference: super::inference::readtimetable,
    extensions: &super::READTIMETABLE_EXTENSIONS,
    integer_capabilities: &super::READTIMETABLE_INTEGER_CAPABILITIES,
    filesystem_read: true,
});
