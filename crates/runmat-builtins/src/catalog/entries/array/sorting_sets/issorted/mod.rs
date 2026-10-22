mod descriptor;
mod documentation;
mod inference;
pub use descriptor::*;

pub const ISSORTED_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::contract::entry(super::contract::Definition {
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "issorted",
        descriptor: &ISSORTED_DESCRIPTOR,
        documentation: documentation::DOCUMENTATION,
        inference: inference::infer,
        compatibility: crate::BuiltinCompatibility::Matlab,
        residency: crate::BuiltinResidencyPolicy::GatherToHost,
        extensions: &ISSORTED_EXTENSIONS,
        integer_capabilities: &ISSORTED_INTEGER_CAPABILITIES,
        integer_audit: None,
    });

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ISSORTED_CATALOG_ENTRY];
