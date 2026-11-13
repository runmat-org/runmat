use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.InitialConditionTarget.wholeModel",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_INITIAL_CONDITION_TARGET_WHOLE_MODEL_DESCRIPTOR,
    inference: super::inference::whole_model,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_INITIAL_CONDITION_TARGET_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
