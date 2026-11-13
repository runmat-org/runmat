mod documentation;

pub(super) const ENTRY: crate::BuiltinCatalogEntry = super::entry(
    crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    "dictionary.subsasgn",
    documentation::DOCUMENTATION,
    &super::DICTIONARY_SUBSASGN_DESCRIPTOR,
    super::inference::dictionary_subsasgn,
    true,
    &super::DICTIONARY_SUBSASGN_INTEGER_CAPABILITIES,
);
