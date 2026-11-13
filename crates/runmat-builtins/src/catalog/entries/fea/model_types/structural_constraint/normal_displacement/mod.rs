use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StructuralConstraint.normalDisplacement",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_STRUCTURAL_CONSTRAINT_NORMAL_DISPLACEMENT_DESCRIPTOR,
    inference: super::inference::normal_displacement,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_STRUCTURAL_CONSTRAINT_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
