mod descriptor;
mod documentation;
mod inference;

pub use descriptor::*;

pub const PARQUET_DATASTORE_ENTRY: crate::BuiltinCatalogEntry =
    super::contract::entry(super::contract::Definition {
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "parquetDatastore",
        category: "io/tabular",
        documentation: documentation::DOCUMENTATION,
        descriptor: &PARQUET_DATASTORE_DESCRIPTOR,
        inference: inference::infer,
        extensions: &[],
        integer_capabilities: &PARQUET_DATASTORE_INTEGER_CAPABILITIES,
        integer_audit: None,
    });
