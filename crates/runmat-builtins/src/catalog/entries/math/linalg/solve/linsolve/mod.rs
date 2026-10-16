mod contract;
mod documentation;

pub use contract::*;

pub const LINSOLVE_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::solve_entry(super::entry::SolveEntrySpec {
        name: "linsolve",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: documentation::DOCUMENTATION,
        descriptor: &LINSOLVE_DESCRIPTOR,
        inference: crate::IdentityInferenceRule::new(super::inference::infer_linsolve),
        extensions: &LINSOLVE_EXTENSIONS,
        integer_capabilities: &LINSOLVE_INTEGER_CAPABILITIES,
        residency: crate::BuiltinResidencyPolicy::Dynamic,
        maturity: crate::BuiltinContractMaturity::DynamicByDesign,
    });
