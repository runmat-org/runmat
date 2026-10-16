mod contract;
mod documentation;

pub use contract::*;

pub const NULL_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::solve_entry(super::entry::SolveEntrySpec {
        name: "null",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: documentation::DOCUMENTATION,
        descriptor: &NULL_DESCRIPTOR,
        inference: crate::IdentityInferenceRule::new(super::inference::infer_null),
        extensions: &NULL_EXTENSIONS,
        integer_capabilities: &NULL_INTEGER_CAPABILITIES,
        residency: crate::BuiltinResidencyPolicy::Dynamic,
        maturity: crate::BuiltinContractMaturity::DynamicByDesign,
    });
