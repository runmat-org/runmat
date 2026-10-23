mod descriptor;
mod documentation;
mod inference;
pub use descriptor::*;

pub const ARGSORT_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::contract::entry(super::contract::Definition {
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "argsort",
        descriptor: &ARGSORT_DESCRIPTOR,
        documentation: documentation::DOCUMENTATION,
        inference: inference::infer,
        compatibility: crate::BuiltinCompatibility::RunMat,
        residency: crate::BuiltinResidencyPolicy::PreserveInputs,
        extensions: &[],
        integer_capabilities: &[],
        integer_audit: Some(&ARGSORT_INTEGER_AUDIT),
    });

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ARGSORT_CATALOG_ENTRY];
