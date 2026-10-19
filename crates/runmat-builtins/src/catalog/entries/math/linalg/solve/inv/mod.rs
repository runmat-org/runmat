mod contract;
mod documentation;

pub use contract::*;

pub const INV_CATALOG_ENTRY: crate::BuiltinCatalogEntry =
    super::entry::solve_entry(super::entry::SolveEntrySpec {
        name: "inv",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: documentation::DOCUMENTATION,
        descriptor: &INV_DESCRIPTOR,
        inference: crate::IdentityInferenceRule::new(super::inference::infer_inv),
        extensions: &INV_EXTENSIONS,
        integer_capabilities: &INV_INTEGER_CAPABILITIES,
        residency: crate::BuiltinResidencyPolicy::Dynamic,
        maturity: crate::BuiltinContractMaturity::DynamicByDesign,
    });
