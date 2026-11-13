use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.ThermalDomain.atReferenceTemperature",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_THERMAL_DOMAIN_AT_REFERENCE_TEMPERATURE_DESCRIPTOR,
    inference: super::inference::at_reference_temperature,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_THERMAL_DOMAIN_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
