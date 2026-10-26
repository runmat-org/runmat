//! Native host delegates numerical semantics to the portable kernel.
#[cfg(test)]
use super::*;
pub(super) use runmat_analysis_execution::execute_state_publication_kernel as execute;
#[cfg(test)]
mod magnetic_tests;
