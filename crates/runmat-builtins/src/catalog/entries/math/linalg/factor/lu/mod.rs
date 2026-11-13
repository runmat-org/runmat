mod contract;
mod documentation;
mod inference;

pub use contract::*;

pub const LU_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::factor_entry(super::entry::FactorEntrySpec {
        name: "lu",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: documentation::DOCUMENTATION,
        descriptor: &LU_DESCRIPTOR,
        inference: crate::IdentityInferenceRule::new(inference::infer),
        extensions: &LU_EXTENSIONS,
        integer_capabilities: &LU_INTEGER_CAPABILITIES,
        residency: crate::BuiltinResidencyPolicy::Dynamic,
        maturity: crate::BuiltinContractMaturity::Complete,
    });

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&LU_CATALOG_ENTRY];
