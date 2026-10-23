mod contract;
mod documentation;
mod inference;

pub use contract::*;

pub const PCG_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::iterative_entry(super::entry::FactorEntrySpec {
        name: "pcg",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: documentation::DOCUMENTATION,
        descriptor: &PCG_DESCRIPTOR,
        inference: crate::IdentityInferenceRule::new(inference::infer),
        extensions: &[],
        integer_capabilities: &[],
        residency: crate::BuiltinResidencyPolicy::Host,
        maturity: crate::BuiltinContractMaturity::Incomplete,
    });

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&PCG_CATALOG_ENTRY];
