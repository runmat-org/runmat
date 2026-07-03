//! Summary statistics builtins.

pub(crate) mod corr;
pub(crate) mod corrcoef;
pub(crate) mod cov;
pub(crate) mod descriptive;
pub(crate) mod distribution_math;
pub(crate) mod distributions;
pub(crate) mod ecdf;
pub(crate) mod hypothesis;
pub(crate) mod mode;
pub(crate) mod normalize;
pub(crate) mod order_stats;

pub use corrcoef::corrcoef_from_tensors;
pub use cov::{cov_from_tensors, CovWeightSpec};
