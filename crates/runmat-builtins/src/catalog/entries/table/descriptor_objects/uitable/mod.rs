mod descriptor;
mod documentation;
mod inference;

pub use descriptor::*;

pub const UITABLE_ENTRY: crate::BuiltinCatalogEntry =
    super::contract::entry(super::contract::Definition {
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "uitable",
        category: "table",
        documentation: documentation::DOCUMENTATION,
        descriptor: &UITABLE_DESCRIPTOR,
        inference: inference::infer,
        extensions: &UITABLE_EXTENSIONS,
        integer_capabilities: &UITABLE_INTEGER_CAPABILITIES,
        integer_audit: None,
    });
