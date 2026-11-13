use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.ElectricalConstraint.floatingConductor",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_ELECTRICAL_CONSTRAINT_FLOATING_CONDUCTOR_DESCRIPTOR,
    inference: super::inference::floating_conductor,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_ELECTRICAL_CONSTRAINT_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
