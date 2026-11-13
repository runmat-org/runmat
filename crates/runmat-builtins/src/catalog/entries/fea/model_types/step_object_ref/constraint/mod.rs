use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StepObjectRef.constraint",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_STEP_OBJECT_REF_CONSTRAINT_DESCRIPTOR,
    inference: super::inference::constraint,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_STEP_OBJECT_REF_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
