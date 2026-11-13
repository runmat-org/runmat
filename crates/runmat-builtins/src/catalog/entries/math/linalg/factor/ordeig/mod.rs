mod contract;
mod documentation;
mod inference;

pub use contract::*;

use crate::*;

pub const ORDEIG_CATALOG_ENTRY: BuiltinCatalogEntry =
    super::entry::owned_host_cpu_factor_entry(super::entry::FactorEntrySpec {
        name: "ordeig",
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: documentation::DOCUMENTATION,
        descriptor: &ORDEIG_DESCRIPTOR,
        inference: IdentityInferenceRule::new(inference::infer),
        extensions: &[],
        integer_capabilities: &[],
        residency: BuiltinResidencyPolicy::Host,
        maturity: BuiltinContractMaturity::Incomplete,
    });

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ORDEIG_CATALOG_ENTRY];
