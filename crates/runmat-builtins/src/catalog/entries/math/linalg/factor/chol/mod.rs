mod contract;
mod documentation;
mod inference;

pub use contract::*;

pub const CHOL_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::factor_entry(super::entry::FactorEntrySpec {
        name: "chol",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: documentation::DOCUMENTATION,
        descriptor: &CHOL_DESCRIPTOR,
        inference: crate::IdentityInferenceRule::new(inference::infer),
        extensions: &CHOL_EXTENSIONS,
        integer_capabilities: &CHOL_INTEGER_CAPABILITIES,
        residency: crate::BuiltinResidencyPolicy::Dynamic,
        maturity: crate::BuiltinContractMaturity::Complete,
    });

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&CHOL_CATALOG_ENTRY];
