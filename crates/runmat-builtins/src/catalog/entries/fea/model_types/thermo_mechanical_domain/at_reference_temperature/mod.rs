use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.ThermoMechanicalDomain.atReferenceTemperature",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_THERMO_MECHANICAL_DOMAIN_AT_REFERENCE_TEMPERATURE_DESCRIPTOR,
    inference: super::inference::at_reference_temperature,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_THERMO_MECHANICAL_DOMAIN_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
