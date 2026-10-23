mod contract;
mod documentation;
mod inference;
pub use contract::*;

pub const GMRES_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::iterative_entry(super::entry::FactorEntrySpec {
        name: "gmres",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: documentation::DOCUMENTATION,
        descriptor: &GMRES_DESCRIPTOR,
        inference: crate::IdentityInferenceRule::new(inference::infer),
        extensions: &[],
        integer_capabilities: &[],
        residency: crate::BuiltinResidencyPolicy::Host,
        maturity: crate::BuiltinContractMaturity::Incomplete,
    });

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&GMRES_CATALOG_ENTRY];
