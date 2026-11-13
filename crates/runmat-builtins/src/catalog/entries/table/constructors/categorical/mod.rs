mod contract;
mod documentation;
mod inference;
mod integer;
pub use contract::*;
pub use integer::*;

pub const CATEGORICAL_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::entry(super::entry::Definition {
        name: "categorical",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        descriptor: &CATEGORICAL_DESCRIPTOR,
        documentation: documentation::DOCUMENTATION,
        inference: inference::infer,
        extensions: &CATEGORICAL_EXTENSIONS,
        integer_capabilities: &CATEGORICAL_INTEGER_CAPABILITIES,
    });
