//! Summary statistics builtins.

mod corrcoef;
mod cov;

pub use corrcoef::corrcoef_from_tensors;
pub use cov::{cov_from_tensors, CovWeightSpec};
