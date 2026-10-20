mod descriptor;
mod documentation;
mod inference;

pub use descriptor::*;

pub const VARTYPE_ENTRY: crate::BuiltinCatalogEntry =
    super::contract::entry(super::contract::Definition {
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "vartype",
        category: "table",
        documentation: documentation::DOCUMENTATION,
        descriptor: &VARTYPE_DESCRIPTOR,
        inference: inference::infer,
        extensions: &[],
        integer_capabilities: &[],
        integer_audit: Some(&VARTYPE_INTEGER_AUDIT),
    });
