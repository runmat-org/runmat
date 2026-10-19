use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.Direction.vector",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_DIRECTION_VECTOR_DESCRIPTOR,
    inference: super::inference::vector,
    integer_capabilities: &super::FEA_DIRECTION_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
