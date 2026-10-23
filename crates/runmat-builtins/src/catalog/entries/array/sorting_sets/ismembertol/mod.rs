mod descriptor;
mod documentation;
mod inference;
pub use descriptor::*;

pub const ISMEMBERTOL_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::contract::entry(super::contract::Definition {
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "ismembertol",
        descriptor: &ISMEMBERTOL_DESCRIPTOR,
        documentation: documentation::DOCUMENTATION,
        inference: inference::infer,
        compatibility: crate::BuiltinCompatibility::Matlab,
        residency: crate::BuiltinResidencyPolicy::Dynamic,
        extensions: &ISMEMBERTOL_EXTENSIONS,
        integer_capabilities: &ISMEMBERTOL_INTEGER_CAPABILITIES,
        integer_audit: None,
    });

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ISMEMBERTOL_CATALOG_ENTRY];
