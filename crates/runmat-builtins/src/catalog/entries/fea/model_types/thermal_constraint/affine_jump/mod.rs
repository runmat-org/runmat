use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.ThermalConstraint.affineJump",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_THERMAL_CONSTRAINT_AFFINE_JUMP_DESCRIPTOR,
    inference: super::inference::affine_jump,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_THERMAL_CONSTRAINT_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
