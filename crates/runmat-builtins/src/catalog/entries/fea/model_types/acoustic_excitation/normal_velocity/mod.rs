use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.AcousticExcitation.normalVelocity",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_ACOUSTIC_EXCITATION_NORMAL_VELOCITY_DESCRIPTOR,
    inference: super::inference::normal_velocity,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_ACOUSTIC_EXCITATION_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
