mod descriptor;
mod documentation;
mod inference;
pub use descriptor::*;

pub const ISMEMBER_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::contract::entry(super::contract::Definition {
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "ismember",
        descriptor: &ISMEMBER_DESCRIPTOR,
        documentation: documentation::DOCUMENTATION,
        inference: inference::infer,
        compatibility: crate::BuiltinCompatibility::Matlab,
        residency: crate::BuiltinResidencyPolicy::PreserveInputs,
        extensions: &[],
        integer_capabilities: &ISMEMBER_INTEGER_CAPABILITIES,
        integer_audit: None,
    });

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ISMEMBER_CATALOG_ENTRY];
