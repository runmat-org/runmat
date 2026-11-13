mod contract;
mod documentation;
mod inference;
mod options;
#[cfg(test)]
mod tests;
pub use options::EigsOption;

pub use contract::*;

pub const EIGS_CATALOG_ENTRY: crate::BuiltinCatalogEntry = {
    let mut entry = super::entry::iterative_entry(super::entry::FactorEntrySpec {
        name: "eigs",
        provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: documentation::DOCUMENTATION,
        descriptor: &EIGS_DESCRIPTOR,
        inference: crate::IdentityInferenceRule::new(inference::infer),
        extensions: &EIGS_EXTENSIONS,
        integer_capabilities: &EIGS_INTEGER_CAPABILITIES,
        residency: crate::BuiltinResidencyPolicy::GatherToHost,
        maturity: crate::BuiltinContractMaturity::Incomplete,
    });
    // The explicit gpuArray compatibility extension retains its gather route;
    // this is not a provider-side iterative eigensolver capability.
    entry.placement.accelerator = crate::BuiltinAcceleratorPolicy::Optional;
    entry.placement.distributed = crate::BuiltinDistributedPolicy::MaterializeArguments;
    entry
};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&EIGS_CATALOG_ENTRY];
