use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.PropertyRegionTarget.cell",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_PROPERTY_REGION_TARGET_CELL_DESCRIPTOR,
    inference: super::inference::cell,
    integer_capabilities: &[],
    integer_audit: Some(&crate::fea_contract::integer::FEA_STRUCTURAL_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
