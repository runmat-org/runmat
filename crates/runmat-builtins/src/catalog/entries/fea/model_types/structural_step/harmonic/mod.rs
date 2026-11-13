use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StructuralStep.harmonic",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_STRUCTURAL_STEP_HARMONIC_DESCRIPTOR,
    inference: super::inference::harmonic,
    integer_capabilities: &[],
    integer_audit: Some(&super::FEA_STRUCTURAL_STEP_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
