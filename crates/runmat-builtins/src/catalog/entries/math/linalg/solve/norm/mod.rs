mod contract;
mod documentation;

pub use contract::*;

pub const NORM_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::solve_entry(super::entry::SolveEntrySpec {
        name: "norm",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: documentation::DOCUMENTATION,
        descriptor: &NORM_DESCRIPTOR,
        inference: crate::IdentityInferenceRule::new(super::inference::infer_norm),
        extensions: &NORM_EXTENSIONS,
        integer_capabilities: &NORM_INTEGER_CAPABILITIES,
        residency: crate::BuiltinResidencyPolicy::Dynamic,
        maturity: crate::BuiltinContractMaturity::DynamicByDesign,
    });
