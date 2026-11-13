use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.MechanicalDamping.undamped",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_MECHANICAL_DAMPING_UNDAMPED_DESCRIPTOR,
    inference: super::inference::undamped,
    integer_capabilities: &[],
    integer_audit: Some(&crate::fea_contract::integer::FEA_STRUCTURAL_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
