mod contract;
mod documentation;
mod inference;
pub use contract::*;

pub const ILU_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::host_factor_entry(super::entry::FactorEntrySpec {
        name: "ilu",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: documentation::DOCUMENTATION,
        descriptor: &ILU_DESCRIPTOR,
        inference: crate::IdentityInferenceRule::new(inference::infer),
        extensions: &[],
        integer_capabilities: &[],
        residency: crate::BuiltinResidencyPolicy::Host,
        maturity: crate::BuiltinContractMaturity::Incomplete,
    });

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ILU_CATALOG_ENTRY];
