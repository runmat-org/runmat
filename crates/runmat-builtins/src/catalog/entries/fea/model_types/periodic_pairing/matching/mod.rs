use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.PeriodicPairing.matching",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_PERIODIC_PAIRING_MATCHING_DESCRIPTOR,
    inference: super::inference::matching,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_PERIODIC_PAIRING_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
