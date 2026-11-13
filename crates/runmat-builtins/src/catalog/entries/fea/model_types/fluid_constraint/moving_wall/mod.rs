use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.FluidConstraint.movingWall",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_FLUID_CONSTRAINT_MOVING_WALL_DESCRIPTOR,
    inference: super::inference::moving_wall,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_FLUID_CONSTRAINT_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
