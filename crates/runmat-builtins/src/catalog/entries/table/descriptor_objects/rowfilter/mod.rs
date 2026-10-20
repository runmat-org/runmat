mod descriptor;
mod documentation;
mod inference;

pub use descriptor::*;

pub const ROWFILTER_ENTRY: crate::BuiltinCatalogEntry =
    super::contract::entry(super::contract::Definition {
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "rowfilter",
        category: "table",
        documentation: documentation::DOCUMENTATION,
        descriptor: &ROWFILTER_DESCRIPTOR,
        inference: inference::infer,
        extensions: &[],
        integer_capabilities: &[],
        integer_audit: Some(&ROWFILTER_INTEGER_AUDIT),
    });
