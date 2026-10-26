use serde::{Deserialize, Serialize};

/// Stable signature API name for the one checked physical descriptor authority.
pub type QuantityDimension = crate::physical::DynamicDimension;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum OperationQuantityRule {
    NotApplicable,
    PreserveInput {
        input: usize,
    },
    RequiredInput {
        input: usize,
        dimension: QuantityDimension,
    },
    FixedOutput {
        output: usize,
        dimension: QuantityDimension,
    },
    CheckedByLowering,
}
