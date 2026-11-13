mod contract;
mod documentation;
mod inference;

pub use contract::*;

pub const SCHUR_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::host_factor_entry(super::entry::FactorEntrySpec {
        name: "schur",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: documentation::DOCUMENTATION,
        descriptor: &SCHUR_DESCRIPTOR,
        inference: crate::IdentityInferenceRule::new(inference::infer),
        extensions: &[],
        integer_capabilities: &[],
        residency: crate::BuiltinResidencyPolicy::Host,
        maturity: crate::BuiltinContractMaturity::Complete,
    });

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&SCHUR_CATALOG_ENTRY];
