use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.ThermalExcitation.heatGeneration",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_THERMAL_EXCITATION_HEAT_GENERATION_DESCRIPTOR,
    inference: super::inference::heat_generation,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_THERMAL_EXCITATION_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
