use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.ElectricalInitialCondition.potential",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_ELECTRICAL_INITIAL_CONDITION_POTENTIAL_DESCRIPTOR,
    inference: super::inference::potential,
    integer_capabilities:
        &crate::fea_contract::model_types::integer::FEA_INITIAL_CONDITION_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
