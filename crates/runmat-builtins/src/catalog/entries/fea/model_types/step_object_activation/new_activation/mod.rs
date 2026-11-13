use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StepObjectActivation.new",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_STEP_OBJECT_ACTIVATION_NEW_ACTIVATION_DESCRIPTOR,
    inference: super::inference::new_activation,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_STEP_OBJECT_ACTIVATION_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
