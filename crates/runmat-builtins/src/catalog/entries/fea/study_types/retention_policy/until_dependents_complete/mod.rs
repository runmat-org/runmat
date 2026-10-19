use crate::fea_contract::{
    catalog_entry::{self, ConstructorEntryDefinition},
    study_types::integer::FEA_OBSERVATION_STRUCTURAL_INTEGER_AUDIT,
};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.RetentionPolicy.untilDependentsComplete",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_RETENTION_POLICY_UNTIL_DEPENDENTS_COMPLETE_DESCRIPTOR,
    inference: super::inference::until_dependents_complete,
    integer_capabilities: &[],
    integer_audit: Some(&FEA_OBSERVATION_STRUCTURAL_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
