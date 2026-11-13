use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.MagneticConstraint.coulombGauge",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_MAGNETIC_CONSTRAINT_COULOMB_GAUGE_DESCRIPTOR,
    inference: super::inference::coulomb_gauge,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_MAGNETIC_CONSTRAINT_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
