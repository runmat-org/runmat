//! Native host delegates numerical semantics to the portable kernel.
#[cfg(test)]
use super::*;
pub(super) use runmat_analysis_execution::execute_structural_linear_kernel as execute;
#[cfg(test)]
mod tests;
