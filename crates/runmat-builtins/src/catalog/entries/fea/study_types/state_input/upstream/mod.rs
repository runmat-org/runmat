use crate::fea_contract::{
    catalog_entry::{self, ConstructorEntryDefinition},
    study_types::integer::FEA_SEQUENCE_STRUCTURAL_INTEGER_AUDIT,
};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StateInput.upstream",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_STATE_INPUT_UPSTREAM_DESCRIPTOR,
    inference: super::inference::upstream,
    integer_capabilities: &[],
    integer_audit: Some(&FEA_SEQUENCE_STRUCTURAL_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
