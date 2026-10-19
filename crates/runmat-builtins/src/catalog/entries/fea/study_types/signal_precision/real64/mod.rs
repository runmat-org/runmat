use crate::fea_contract::{
    catalog_entry::{self, ConstructorEntryDefinition},
    study_types::integer::FEA_OBSERVATION_STRUCTURAL_INTEGER_AUDIT,
};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.SignalPrecision.real64",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_SIGNAL_PRECISION_REAL64_DESCRIPTOR,
    inference: super::inference::real64,
    integer_capabilities: &[],
    integer_audit: Some(&FEA_OBSERVATION_STRUCTURAL_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
