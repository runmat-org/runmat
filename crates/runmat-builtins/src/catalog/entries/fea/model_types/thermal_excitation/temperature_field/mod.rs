use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.ThermalExcitation.temperatureField",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_THERMAL_EXCITATION_TEMPERATURE_FIELD_DESCRIPTOR,
    inference: super::inference::temperature_field,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_THERMAL_EXCITATION_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
