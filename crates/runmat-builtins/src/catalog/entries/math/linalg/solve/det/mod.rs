mod contract;
mod documentation;

pub use contract::*;

pub const DET_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::solve_entry(super::entry::SolveEntrySpec {
        name: "det",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: documentation::DOCUMENTATION,
        descriptor: &DET_DESCRIPTOR,
        inference: crate::IdentityInferenceRule::new(super::inference::infer_det),
        extensions: &DET_EXTENSIONS,
        integer_capabilities: &DET_INTEGER_CAPABILITIES,
        residency: crate::BuiltinResidencyPolicy::Dynamic,
        maturity: crate::BuiltinContractMaturity::DynamicByDesign,
    });
