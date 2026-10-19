//! Atomic reservations from the authoritative session RNG stream.
//!
//! Producers allocate and check cancellation before reserving. A reservation
//! advances the stream once, then numerical work runs without its borrow or
//! mutex. Abandoning that work does not rewind draws another call may use.

use super::{turn::RandomTurn, RandomState};
use crate::context::RuntimeContext;
use crate::execution::ExecutionServiceError;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(super) struct DrawReservation {
    pub(super) first_state: RandomState,
    pub(super) following_state: RandomState,
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
    turn: &mut RandomTurn,
    draws: u64,
    label: &str,
) -> BuiltinResult<DrawReservation> {
    let owner = turn.owner().cloned();
    turn.with_state(label, |state| {
        if draws == 0 {
            return Ok(DrawReservation {
                first_state: state.state.clone(),
                following_state: state.state.clone(),
            });
        }
        state.state.prepare_mutation(owner.as_ref())?;
        let mut first_state = state.state.clone();
        first_state.prepare_mutation(owner.as_ref())?;
        state.state.advance_uniforms(draws, owner.as_ref())?;
        Ok(DrawReservation {
            first_state,
            following_state: state.state.clone(),
        })
    })?
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
