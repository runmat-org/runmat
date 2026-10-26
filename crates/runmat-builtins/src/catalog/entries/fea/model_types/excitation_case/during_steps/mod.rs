use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.ExcitationCase.duringSteps",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_EXCITATION_CASE_DURING_STEPS_DESCRIPTOR,
    inference: super::inference::during_steps,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_EXCITATION_CASE_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
