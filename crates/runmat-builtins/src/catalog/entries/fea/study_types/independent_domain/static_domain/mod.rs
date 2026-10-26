use crate::fea_contract::{
    catalog_entry::{self, ConstructorEntryDefinition},
    study_types::integer::FEA_SEQUENCE_STRUCTURAL_INTEGER_AUDIT,
};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.IndependentDomain.staticDomain",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_INDEPENDENT_DOMAIN_STATIC_DOMAIN_DESCRIPTOR,
    inference: super::inference::static_domain,
    integer_capabilities: &[],
    integer_audit: Some(&FEA_SEQUENCE_STRUCTURAL_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
