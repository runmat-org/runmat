mod contract;
mod documentation;

pub use contract::*;

pub const PINV_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::solve_entry(super::entry::SolveEntrySpec {
        name: "pinv",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: documentation::DOCUMENTATION,
        descriptor: &PINV_DESCRIPTOR,
        inference: crate::IdentityInferenceRule::new(super::inference::infer_pinv),
        extensions: &PINV_EXTENSIONS,
        integer_capabilities: &PINV_INTEGER_CAPABILITIES,
        residency: crate::BuiltinResidencyPolicy::Dynamic,
        maturity: crate::BuiltinContractMaturity::DynamicByDesign,
    });
