use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.AcousticBoundary.rigidWall",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_ACOUSTIC_BOUNDARY_RIGID_WALL_DESCRIPTOR,
    inference: super::inference::rigid_wall,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_ACOUSTIC_BOUNDARY_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
