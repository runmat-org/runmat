mod contract;
mod documentation;
mod inference;

pub use contract::*;

pub const EIGS_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::factor_entry(super::entry::FactorEntrySpec {
        name: "eigs",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: documentation::DOCUMENTATION,
        descriptor: &EIGS_DESCRIPTOR,
        inference: crate::IdentityInferenceRule::new(inference::infer),
        extensions: &EIGS_EXTENSIONS,
        integer_capabilities: &EIGS_INTEGER_CAPABILITIES,
        residency: crate::BuiltinResidencyPolicy::GatherToHost,
        maturity: crate::BuiltinContractMaturity::Incomplete,
    });
