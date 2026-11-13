mod descriptor;
mod documentation;
mod inference;
pub use descriptor::*;

pub const SETDIFF_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::contract::entry(super::contract::Definition {
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "setdiff",
        descriptor: &SETDIFF_DESCRIPTOR,
        documentation: documentation::DOCUMENTATION,
        inference: inference::infer,
        compatibility: crate::BuiltinCompatibility::Matlab,
        residency: crate::BuiltinResidencyPolicy::PreserveInputs,
        extensions: &[],
        integer_capabilities: &SETDIFF_INTEGER_CAPABILITIES,
        integer_audit: None,
    });

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&SETDIFF_CATALOG_ENTRY];
