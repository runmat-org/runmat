mod contract;
mod documentation;

pub use contract::*;

pub const VECNORM_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::solve_entry(super::entry::SolveEntrySpec {
        name: "vecnorm",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: documentation::DOCUMENTATION,
        descriptor: &VECNORM_DESCRIPTOR,
        inference: crate::IdentityInferenceRule::new(super::inference::infer_vecnorm),
        extensions: &VECNORM_EXTENSIONS,
        integer_capabilities: &VECNORM_INTEGER_CAPABILITIES,
        residency: crate::BuiltinResidencyPolicy::Dynamic,
        maturity: crate::BuiltinContractMaturity::DynamicByDesign,
    });
