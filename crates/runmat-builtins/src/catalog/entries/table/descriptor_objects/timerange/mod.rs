mod descriptor;
mod documentation;
mod inference;

pub use descriptor::*;

pub const TIMERANGE_ENTRY: crate::BuiltinCatalogEntry =
    super::contract::entry(super::contract::Definition {
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "timerange",
        category: "table",
        documentation: documentation::DOCUMENTATION,
        descriptor: &TIMERANGE_DESCRIPTOR,
        inference: inference::infer,
        extensions: &TIMERANGE_EXTENSIONS,
        integer_capabilities: &TIMERANGE_INTEGER_CAPABILITIES,
        integer_audit: None,
    });
