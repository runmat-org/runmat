mod contract;
mod documentation;

pub use contract::*;

pub const RANK_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::solve_entry(super::entry::SolveEntrySpec {
        name: "rank",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: documentation::DOCUMENTATION,
        descriptor: &RANK_DESCRIPTOR,
        inference: crate::IdentityInferenceRule::new(super::inference::infer_rank),
        extensions: &RANK_EXTENSIONS,
        integer_capabilities: &RANK_INTEGER_CAPABILITIES,
        residency: crate::BuiltinResidencyPolicy::Dynamic,
        maturity: crate::BuiltinContractMaturity::DynamicByDesign,
    });
