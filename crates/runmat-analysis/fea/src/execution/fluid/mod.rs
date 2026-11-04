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
pub use fields::{fluid_native_field_definition_inventory, fluid_native_field_definitions};
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
    FluidIntervalEvidence, FluidTransientBlock, FluidTransientBlockReservation,
    FluidTransientBlockResult,
};
pub use time_schedule::FluidTransientSchedule;
pub use transient_workload::{
    FluidTransientSolverResult, FluidTransientWorkload, PreparedFluidTransient,
};

pub use result::{FluidResultOnSource, FluidRetainedOriginal, FluidSteadySolverResult};
pub use workload::{FluidCoupling, FluidSteadyWorkload};

use super::{AnalysisExecutionError, AnalysisExecutionResult};
use runmat_canonical_codec::CanonicalLimits;

const WORKLOAD_PREFIX: &[u8] = b"runmat-steady-fluid-workload/v1\0";
const RESULT_PREFIX: &[u8] = b"runmat-steady-fluid-retained-result/v1\0";
/// Codec limits of fluid workloads and retained results. They are structural sanity bounds, not
/// size caps: a retained result grows with cells, faces and retained steps, and its memory is
/// admitted by the phase resource plan before it is built or decoded.
const LIMITS: CanonicalLimits = CanonicalLimits::new(
    MAX_PAYLOAD_BYTES,
    crate::address_space_power_of_two(36),
    65_536,
    64,
);
const MAX_PAYLOAD_BYTES: usize = crate::address_space_power_of_two(44);

fn invalid(error: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(error.to_string())
}

fn fluid_error(
    error: crate::physics::cfd::canonical::CanonicalFluidError,
) -> AnalysisExecutionError {
    use crate::physics::cfd::canonical::CanonicalFluidError;
    match error {
        CanonicalFluidError::Accounting(cause) => AnalysisExecutionError::Usage(cause),
        CanonicalFluidError::Cancelled => AnalysisExecutionError::Cancelled,
        other => invalid(other),
    }
}

fn fluid_structure_error(
    error: crate::physics::fluid_structure::FluidStructureError,
) -> AnalysisExecutionError {
    use crate::physics::fluid_structure::FluidStructureError;
    match error {
        FluidStructureError::Flow(cause) => fluid_error(cause),
        FluidStructureError::Cancelled => AnalysisExecutionError::Cancelled,
        other => invalid(other),
    }
}
