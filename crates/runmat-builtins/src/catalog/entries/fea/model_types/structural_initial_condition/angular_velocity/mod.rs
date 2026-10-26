use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StructuralInitialCondition.angularVelocity",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_STRUCTURAL_INITIAL_CONDITION_ANGULAR_VELOCITY_DESCRIPTOR,
    inference: super::inference::angular_velocity,
    integer_capabilities:
        &crate::fea_contract::model_types::integer::FEA_INITIAL_CONDITION_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
