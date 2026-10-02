//! Portable original steady-fluid authority. Not a scheduler plan or publication gate.
//! Cold validation evaluates original retained fields; it never invokes the solver.
mod fields;
mod profile;
mod result;
pub use fields::fluid_native_field_definition_inventory;
#[cfg(test)]
mod tests;
mod workload;

pub use result::FluidSteadySolverResult;
pub use workload::FluidSteadyWorkload;

use super::{AnalysisExecutionError, AnalysisExecutionResult};
use runmat_canonical_codec::CanonicalLimits;

const WORKLOAD_PREFIX: &[u8] = b"runmat-steady-fluid-workload/v1\0";
const RESULT_PREFIX: &[u8] = b"runmat-steady-fluid-retained-result/v1\0";
const LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 2_000_000, 65_536, 64);
const MAX_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;

fn invalid(error: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(error.to_string())
}
