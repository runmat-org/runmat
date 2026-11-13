mod contract;
mod documentation;
mod inference;
mod integer;
pub use contract::*;
pub use integer::*;

pub const DICTIONARY_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::entry(super::entry::Definition {
        name: "dictionary",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        descriptor: &DICTIONARY_DESCRIPTOR,
        documentation: documentation::DOCUMENTATION,
        inference: inference::infer,
        extensions: &DICTIONARY_EXTENSIONS,
        integer_capabilities: &DICTIONARY_INTEGER_CAPABILITIES,
    });
