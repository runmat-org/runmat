use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StructuralExcitation.traction",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_STRUCTURAL_EXCITATION_TRACTION_DESCRIPTOR,
    inference: super::inference::traction,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_STRUCTURAL_EXCITATION_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
