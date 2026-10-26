use crate::fea_contract::{
    catalog_entry::{self, ConstructorEntryDefinition},
    study_types::integer::FEA_INDEPENDENT_DOMAIN_INTEGER_CAPABILITIES,
};
use crate::*;

mod documentation;

pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.IndependentDomain.eigenvalue",
    documentation: documentation::DOCUMENTATION,
    descriptor: &super::FEA_INDEPENDENT_DOMAIN_EIGENVALUE_DESCRIPTOR,
    inference: super::inference::eigenvalue,
    integer_capabilities: &FEA_INDEPENDENT_DOMAIN_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
