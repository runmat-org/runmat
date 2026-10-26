use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.MechanicalDamping.rayleigh",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_MECHANICAL_DAMPING_RAYLEIGH_DESCRIPTOR,
    inference: super::inference::rayleigh,
    integer_capabilities: &[],
    integer_audit: Some(&crate::fea_contract::integer::FEA_STRUCTURAL_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
