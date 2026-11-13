mod contract;
mod documentation;
mod inference;

pub use contract::*;

pub const QZ_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::owned_host_cpu_factor_entry(super::entry::FactorEntrySpec {
        name: "qz",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: documentation::DOCUMENTATION,
        descriptor: &QZ_DESCRIPTOR,
        inference: crate::IdentityInferenceRule::new(inference::infer),
        extensions: &[],
        integer_capabilities: &[],
        residency: crate::BuiltinResidencyPolicy::Host,
        maturity: crate::BuiltinContractMaturity::Incomplete,
    });

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&QZ_CATALOG_ENTRY];
