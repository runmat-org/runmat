//! Public variable-constructor contract and shared Bayesian-variable identities.
mod contract;
mod documentation;
mod entry;
mod inference;
mod options;
#[cfg(test)]
mod tests;

pub use contract::{
    ERROR_FLOW as OPTIMIZABLE_VARIABLE_ERROR_FLOW,
    ERROR_INVALID_ARGUMENT as OPTIMIZABLE_VARIABLE_ERROR_INVALID_ARGUMENT,
    ERROR_INVALID_OPTION as OPTIMIZABLE_VARIABLE_ERROR_INVALID_OPTION,
    ERROR_INVALID_RANGE as OPTIMIZABLE_VARIABLE_ERROR_INVALID_RANGE,
    INTEGER_CAPABILITIES as OPTIMIZABLE_VARIABLE_INTEGER_CAPABILITIES,
    OPTIMIZABLE_VARIABLE_DESCRIPTOR, OPTIMIZABLE_VARIABLE_INTEGER_RANGE_EXTENSION,
};
pub use entry::OPTIMIZABLE_VARIABLE_CATALOG_ENTRY;
pub use options::{
    OptimizableVariableOption, OptimizationVariableTransform, OptimizationVariableType,
};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&OPTIMIZABLE_VARIABLE_CATALOG_ENTRY];
