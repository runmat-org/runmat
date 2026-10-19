use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.AcousticInitialCondition.pressure",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_ACOUSTIC_INITIAL_CONDITION_PRESSURE_DESCRIPTOR,
    inference: super::inference::pressure,
    integer_capabilities:
        &crate::fea_contract::model_types::integer::FEA_INITIAL_CONDITION_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
