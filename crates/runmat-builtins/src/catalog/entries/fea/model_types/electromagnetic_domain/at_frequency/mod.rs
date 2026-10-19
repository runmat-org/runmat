use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.ElectromagneticDomain.atFrequency",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_ELECTROMAGNETIC_DOMAIN_AT_FREQUENCY_DESCRIPTOR,
    inference: super::inference::at_frequency,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_ELECTROMAGNETIC_DOMAIN_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
