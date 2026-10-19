use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.CoordinateFrame.cylindrical",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_COORDINATE_FRAME_CYLINDRICAL_DESCRIPTOR,
    inference: super::inference::cylindrical,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_COORDINATE_FRAME_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
