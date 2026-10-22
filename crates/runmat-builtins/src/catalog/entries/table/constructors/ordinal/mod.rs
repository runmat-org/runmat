mod contract;
mod documentation;
mod inference;
mod integer;
pub use contract::*;
pub use integer::*;

pub const ORDINAL_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::entry(super::entry::Definition {
        name: "ordinal",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        descriptor: &ORDINAL_DESCRIPTOR,
        documentation: documentation::DOCUMENTATION,
        inference: inference::infer,
        extensions: &ORDINAL_EXTENSIONS,
        integer_capabilities: &ORDINAL_INTEGER_CAPABILITIES,
    });
