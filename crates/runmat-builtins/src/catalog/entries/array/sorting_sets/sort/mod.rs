mod descriptor;
mod documentation;
mod inference;
pub use descriptor::*;

pub const SORT_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::contract::entry(super::contract::Definition {
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "sort",
        descriptor: &SORT_DESCRIPTOR,
        documentation: documentation::DOCUMENTATION,
        inference: inference::infer,
        compatibility: crate::BuiltinCompatibility::Matlab,
        residency: crate::BuiltinResidencyPolicy::PreserveInputs,
        extensions: &[],
        integer_capabilities: &SORT_INTEGER_CAPABILITIES,
        integer_audit: None,
    });

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&SORT_CATALOG_ENTRY];
