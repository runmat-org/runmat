mod descriptor;
mod documentation;
mod inference;
pub use descriptor::*;

pub const ISSORTEDROWS_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::contract::entry(super::contract::Definition {
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "issortedrows",
        descriptor: &ISSORTEDROWS_DESCRIPTOR,
        documentation: documentation::DOCUMENTATION,
        inference: inference::infer,
        compatibility: crate::BuiltinCompatibility::Matlab,
        residency: crate::BuiltinResidencyPolicy::GatherToHost,
        extensions: &ISSORTEDROWS_EXTENSIONS,
        integer_capabilities: &ISSORTEDROWS_INTEGER_CAPABILITIES,
        integer_audit: None,
    });

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ISSORTEDROWS_CATALOG_ENTRY];
