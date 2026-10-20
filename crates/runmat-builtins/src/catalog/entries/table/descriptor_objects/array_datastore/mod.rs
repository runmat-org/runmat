mod descriptor;
mod documentation;
mod inference;

pub use descriptor::*;

pub const ARRAY_DATASTORE_ENTRY: crate::BuiltinCatalogEntry =
    super::contract::entry(super::contract::Definition {
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "arrayDatastore",
        category: "io/tabular",
        documentation: documentation::DOCUMENTATION,
        descriptor: &ARRAY_DATASTORE_DESCRIPTOR,
        inference: inference::infer,
        extensions: &ARRAY_DATASTORE_EXTENSIONS,
        integer_capabilities: &ARRAY_DATASTORE_INTEGER_CAPABILITIES,
        integer_audit: None,
    });
