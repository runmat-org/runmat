use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.PropertyRegionTarget.solid",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_PROPERTY_REGION_TARGET_SOLID_DESCRIPTOR,
    inference: super::inference::solid,
    integer_capabilities: &[],
    integer_audit: Some(&crate::fea_contract::integer::FEA_STRUCTURAL_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
