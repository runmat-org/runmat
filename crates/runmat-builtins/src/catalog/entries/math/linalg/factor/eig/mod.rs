mod contract;
mod documentation;
mod inference;

pub use contract::*;

pub const EIG_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::factor_entry(super::entry::FactorEntrySpec {
        name: "eig",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: documentation::DOCUMENTATION,
        descriptor: &EIG_DESCRIPTOR,
        inference: crate::IdentityInferenceRule::new(inference::infer),
        extensions: &EIG_EXTENSIONS,
        integer_capabilities: &EIG_INTEGER_CAPABILITIES,
        residency: crate::BuiltinResidencyPolicy::Dynamic,
        maturity: crate::BuiltinContractMaturity::Complete,
    });

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&EIG_CATALOG_ENTRY];
