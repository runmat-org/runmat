//! Atomic reservations from the authoritative session RNG stream.
//!
//! Producers allocate and check cancellation before reserving. A reservation
//! advances the stream once, then numerical work runs without its borrow or
//! mutex. Abandoning that work does not rewind draws another call may use.

use super::{advance_state, with_rng_state_in, RngAlgorithm};
use crate::context::RuntimeContext;
use crate::execution::ExecutionServiceError;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(super) struct DrawReservation {
    pub(super) first_state: u64,
    pub(super) following_state: u64,
}

pub(super) fn checked_length(len: usize, label: &str) -> BuiltinResult<u64> {
    u64::try_from(len).map_err(|source| {
        build_runtime_error(format!("{label}: draw count exceeds the stream range"))
            .with_builtin(label)
            .with_source(source)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
    })
}

pub(super) fn output_storage(len: usize, label: &str) -> BuiltinResult<Vec<f64>> {
    let mut output = Vec::new();
    output.try_reserve_exact(len).map_err(|source| {
        build_runtime_error(format!("{label}: cannot reserve random output storage"))
            .with_builtin(label)
            .with_source(source)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
    })?;
    Ok(output)
}

pub(super) fn reserve(
    draws: u64,
    label: &str,
    context: Option<&RuntimeContext>,
) -> BuiltinResult<DrawReservation> {
    with_rng_state_in(context, label, |state| {
        let first_state = state.state;
        let following_state = match state.algorithm {
            RngAlgorithm::RunMatLcg => advance_state(first_state, draws),
        };
        state.state = following_state;
        DrawReservation {
            first_state,
            following_state,
        }
    })
}

pub(super) fn service_error(
    label: &str,
    context: Option<&RuntimeContext>,
    source: ExecutionServiceError,
) -> RuntimeError {
    let cancelled = matches!(source, ExecutionServiceError::Cancelled);
    let mut error = build_runtime_error(format!("{label}: random generation was interrupted"))
        .with_builtin(label)
        .with_source(source)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if cancelled {
        let namespace = context
            .map(RuntimeContext::error_namespace)
            .unwrap_or_else(|| "RunMat".to_string());
        error = error.with_identifier(format!("{namespace}:Cancelled"));
    }
    error.build()
}
