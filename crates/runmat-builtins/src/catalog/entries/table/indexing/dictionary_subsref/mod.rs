mod documentation;

pub(super) const ENTRY: crate::BuiltinCatalogEntry = super::entry(
    crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    "dictionary.subsref",
    documentation::DOCUMENTATION,
    &super::DICTIONARY_SUBSREF_DESCRIPTOR,
    super::inference::dictionary_subsref,
    false,
    &super::DICTIONARY_SUBSREF_INTEGER_CAPABILITIES,
);
