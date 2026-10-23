mod contract;
mod documentation;
mod inference;

pub use contract::*;

pub const SVD_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::factor_entry(super::entry::FactorEntrySpec {
        name: "svd",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: documentation::DOCUMENTATION,
        descriptor: &SVD_DESCRIPTOR,
        inference: crate::IdentityInferenceRule::new(inference::infer),
        extensions: &SVD_EXTENSIONS,
        integer_capabilities: &SVD_INTEGER_CAPABILITIES,
        residency: crate::BuiltinResidencyPolicy::GatherToHost,
        maturity: crate::BuiltinContractMaturity::Complete,
    });

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&SVD_CATALOG_ENTRY];
