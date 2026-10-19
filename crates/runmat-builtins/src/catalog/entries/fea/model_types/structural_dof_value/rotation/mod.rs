use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StructuralDofValue.rotation",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_STRUCTURAL_DOF_VALUE_ROTATION_DESCRIPTOR,
    inference: super::inference::rotation,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_STRUCTURAL_DOF_VALUE_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
