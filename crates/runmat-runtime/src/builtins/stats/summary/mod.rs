//! Summary statistics builtins.

pub(crate) mod corrcoef;
pub(crate) mod cov;
pub(crate) mod mode;

pub use corrcoef::corrcoef_from_tensors;
pub use cov::{cov_from_tensors, CovWeightSpec};
