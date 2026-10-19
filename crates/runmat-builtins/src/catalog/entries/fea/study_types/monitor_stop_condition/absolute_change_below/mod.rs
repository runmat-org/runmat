use crate::fea_contract::{
    catalog_entry::{self, ConstructorEntryDefinition},
    study_types::integer::FEA_OBSERVATION_NUMERIC_INTEGER_CAPABILITIES,
};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.MonitorStopCondition.absoluteChangeBelow",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_MONITOR_STOP_CONDITION_ABSOLUTE_CHANGE_BELOW_DESCRIPTOR,
    inference: super::inference::absolute_change_below,
    integer_capabilities: &FEA_OBSERVATION_NUMERIC_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
