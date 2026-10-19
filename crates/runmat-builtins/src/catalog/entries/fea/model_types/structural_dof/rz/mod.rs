use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StructuralDof.rz",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_STRUCTURAL_DOF_RZ_DESCRIPTOR,
    inference: super::inference::rz,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_STRUCTURAL_DOF_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
