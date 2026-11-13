use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.AcousticExcitation.pressure",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_ACOUSTIC_EXCITATION_PRESSURE_DESCRIPTOR,
    inference: super::inference::pressure,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_ACOUSTIC_EXCITATION_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
