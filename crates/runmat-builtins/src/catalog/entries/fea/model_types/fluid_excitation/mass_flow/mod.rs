use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.FluidExcitation.massFlow",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_FLUID_EXCITATION_MASS_FLOW_DESCRIPTOR,
    inference: super::inference::mass_flow,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_FLUID_EXCITATION_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
