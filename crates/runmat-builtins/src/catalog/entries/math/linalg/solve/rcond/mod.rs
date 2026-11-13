mod contract;
mod documentation;

pub use contract::*;

pub const RCOND_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::solve_entry(super::entry::SolveEntrySpec {
        name: "rcond",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: documentation::DOCUMENTATION,
        descriptor: &RCOND_DESCRIPTOR,
        inference: crate::IdentityInferenceRule::new(super::inference::infer_rcond),
        extensions: &RCOND_EXTENSIONS,
        integer_capabilities: &RCOND_INTEGER_CAPABILITIES,
        residency: crate::BuiltinResidencyPolicy::Dynamic,
        maturity: crate::BuiltinContractMaturity::DynamicByDesign,
    });
