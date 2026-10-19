use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.ConjugateHeatTransferInterface.resistance",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_CONJUGATE_HEAT_TRANSFER_INTERFACE_RESISTANCE_DESCRIPTOR,
    inference: super::inference::resistance,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_CONJUGATE_HEAT_TRANSFER_INTERFACE_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
