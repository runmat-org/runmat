use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StructuralExcitation.moment",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_STRUCTURAL_EXCITATION_MOMENT_DESCRIPTOR,
    inference: super::inference::moment,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_STRUCTURAL_EXCITATION_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
