mod contract;
mod documentation;

pub use contract::*;

pub const RREF_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::solve_entry(super::entry::SolveEntrySpec {
        name: "rref",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: documentation::DOCUMENTATION,
        descriptor: &RREF_DESCRIPTOR,
        inference: crate::IdentityInferenceRule::new(super::inference::infer_rref),
        extensions: &RREF_EXTENSIONS,
        integer_capabilities: &RREF_INTEGER_CAPABILITIES,
        residency: crate::BuiltinResidencyPolicy::Dynamic,
        maturity: crate::BuiltinContractMaturity::DynamicByDesign,
    });
