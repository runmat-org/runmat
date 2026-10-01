//! Callback-driven Bayesian optimization contract.
mod contract;
mod documentation;
mod entry;
mod inference;
mod options;
#[cfg(test)]
mod tests;

pub use contract::{
    BAYESOPT_DESCRIPTOR, ERROR_FLOW as BAYESOPT_ERROR_FLOW,
    ERROR_INVALID_ARGUMENT as BAYESOPT_ERROR_INVALID_ARGUMENT,
    ERROR_OBJECTIVE as BAYESOPT_ERROR_OBJECTIVE, ERROR_UNSUPPORTED as BAYESOPT_ERROR_UNSUPPORTED,
};
pub use entry::BAYESOPT_CATALOG_ENTRY;
pub use options::{BayesoptAcquisition, BayesoptOption};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&BAYESOPT_CATALOG_ENTRY];
