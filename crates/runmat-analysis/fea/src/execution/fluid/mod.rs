//! Portable original steady-fluid authority. Not a scheduler plan or publication gate.
//! Cold validation evaluates original retained fields; it never invokes the solver.
mod admitted;
mod fields;
mod receipt;
pub use receipt::{
    fluid_receipt_encoding_profile, FluidNumericalReceipt, FluidNumericalReceiptArtifact,
    FluidSolverProvenance, FLUID_RECEIPT_MAX_ENCODED_BYTES,
};
mod profile;
mod result;
pub use fields::{
    fluid_native_field_definition_inventory, fluid_native_field_definitions,
};
mod checkpoint;
mod source;
pub use checkpoint::{
    FluidTransientBlockWorkload, FluidTransientCheckpoint, PreparedFluidTransientContinuation,
};
#[cfg(test)]
pub(crate) mod tests;
mod time_block;
mod time_schedule;
mod transient_workload;
mod workload;
pub use time_block::{
    FluidTransientBlock, FluidTransientBlockReservation, FluidTransientBlockResult,
};
pub use time_schedule::FluidTransientSchedule;
pub use transient_workload::{
    FluidTransientSolverResult, FluidTransientWorkload, PreparedFluidTransient,
};

pub use result::{FluidRetainedOriginal, FluidSteadySolverResult};
pub use workload::{FluidCoupling, FluidSteadyWorkload};

use super::{AnalysisExecutionError, AnalysisExecutionResult};
use runmat_canonical_codec::CanonicalLimits;

const WORKLOAD_PREFIX: &[u8] = b"runmat-steady-fluid-workload/v1\0";
const RESULT_PREFIX: &[u8] = b"runmat-steady-fluid-retained-result/v1\0";
const LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 2_000_000, 65_536, 64);
const MAX_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;

fn invalid(error: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(error.to_string())
}
