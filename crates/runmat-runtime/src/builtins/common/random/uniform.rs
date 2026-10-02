//! Contiguous uniform draws from the session's existing RunMat LCG stream.
//!
//! Reservation is the stream's linearization point. A cancelled or abandoned
//! producer consumes its entire reserved range; it never rewinds a stream that
//! another call may already have advanced. No RNG guard is held across work or
//! an execution-service turn.

use super::{advance_state, next_uniform_state, with_rng_state_in, RngAlgorithm};
use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

struct UniformReservation {
    first_state: u64,
    following_state: u64,
    algorithm: RngAlgorithm,
}

fn checked_length(len: usize, label: &str) -> BuiltinResult<u64> {
    u64::try_from(len).map_err(|source| {
        build_runtime_error(format!(
            "{label}: uniform draw count exceeds the stream range"
        ))
        .with_builtin(label)
        .with_source(source)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
    })
}

fn output_storage(len: usize, label: &str) -> BuiltinResult<Vec<f64>> {
    let mut output = Vec::new();
    output.try_reserve_exact(len).map_err(|source| {
        build_runtime_error(format!("{label}: cannot reserve uniform output storage"))
            .with_builtin(label)
            .with_source(source)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
    })?;
    Ok(output)
}

fn reserve(
    len: u64,
    label: &str,
    context: Option<&RuntimeContext>,
) -> BuiltinResult<UniformReservation> {
    with_rng_state_in(context, label, |state| {
        let first_state = state.state;
        let following_state = match state.algorithm {
            RngAlgorithm::RunMatLcg => advance_state(first_state, len),
        };
        state.state = following_state;
        UniformReservation {
            first_state,
            following_state,
            algorithm: state.algorithm,
        }
    })
}

fn next_uniform(reservation: &UniformReservation, local: &mut u64) -> f64 {
    match reservation.algorithm {
        RngAlgorithm::RunMatLcg => next_uniform_state(local),
    }
}

fn service_error(
    label: &str,
    context: Option<&RuntimeContext>,
    source: ExecutionServiceError,
) -> RuntimeError {
    let cancelled = matches!(source, ExecutionServiceError::Cancelled);
    let mut error = build_runtime_error(format!("{label}: uniform generation was interrupted"))
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

/// The synchronous caller and cooperative caller use the same reservation and
/// word-to-double conversion. A call's draws remain contiguous even when
/// another invocation advances the stream while this call is suspended.
pub(crate) fn generate_uniform(len: usize, label: &str) -> BuiltinResult<Vec<f64>> {
    let count = checked_length(len, label)?;
    let mut output = output_storage(len, label)?;
    let reservation = reserve(count, label, None)?;
    let mut local = reservation.first_state;
    for _ in 0..len {
        output.push(next_uniform(&reservation, &mut local));
    }
    debug_assert_eq!(local, reservation.following_state);
    Ok(output)
}

/// Generate one reserved range with the invocation's shared CPU-turn budget.
/// Both resource admission and the first cancellation check precede reservation.
/// If a turn fails after reservation, its original typed service cause is kept;
/// the unused suffix is consumed, not handed to a later call.
pub(crate) async fn generate_uniform_cooperative(
    len: usize,
    label: &str,
    context: Option<&RuntimeContext>,
    budget: &mut CpuCooperationBudget,
) -> BuiltinResult<Vec<f64>> {
    budget
        .checkpoint(0)
        .await
        .map_err(|source| service_error(label, context, source))?;
    let count = checked_length(len, label)?;
    let mut output = output_storage(len, label)?;
    budget
        .checkpoint(0)
        .await
        .map_err(|source| service_error(label, context, source))?;
    let reservation = reserve(count, label, context)?;
    let mut local = reservation.first_state;
    for _ in 0..len {
        output.push(next_uniform(&reservation, &mut local));
        budget
            .checkpoint(1)
            .await
            .map_err(|source| service_error(label, context, source))?;
    }
    debug_assert_eq!(local, reservation.following_state);
    Ok(output)
}

#[cfg(test)]
#[path = "uniform/tests.rs"]
mod tests;
