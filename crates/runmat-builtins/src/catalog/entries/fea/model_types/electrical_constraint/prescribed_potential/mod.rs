use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.ElectricalConstraint.prescribedPotential",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_ELECTRICAL_CONSTRAINT_PRESCRIBED_POTENTIAL_DESCRIPTOR,
    inference: super::inference::prescribed_potential,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_ELECTRICAL_CONSTRAINT_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
