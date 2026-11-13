mod contract;
mod documentation;
mod inference;
mod integer;
pub use contract::*;
pub use integer::*;

pub const TABLE_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::entry(super::entry::Definition {
        name: "table",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        descriptor: &TABLE_DESCRIPTOR,
        documentation: documentation::DOCUMENTATION,
        inference: inference::infer,
        extensions: &TABLE_EXTENSIONS,
        integer_capabilities: &TABLE_INTEGER_CAPABILITIES,
    });
