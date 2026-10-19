use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.ActivationLifecycle.continueFrom",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_ACTIVATION_LIFECYCLE_CONTINUE_FROM_DESCRIPTOR,
    inference: super::inference::continue_from,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_ACTIVATION_LIFECYCLE_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
