//! Contiguous uniform draws from the session's existing RunMat LCG stream.
//!
//! Reservation is the stream's linearization point. A cancelled or abandoned
//! producer consumes its entire reserved range; it never rewinds a stream that
//! another call may already have advanced. No RNG guard is held across work or
//! an execution-service turn.

use super::next_uniform_state;
use super::reservation::{checked_length, output_storage, reserve, service_error};
use crate::context::RuntimeContext;
use crate::execution::CpuCooperationBudget;
use crate::BuiltinResult;

/// The synchronous caller and cooperative caller use the same reservation and
/// word-to-double conversion. A call's draws remain contiguous even when
/// another invocation advances the stream while this call is suspended.
pub(crate) fn generate_uniform(len: usize, label: &str) -> BuiltinResult<Vec<f64>> {
    let count = checked_length(len, label)?;
    let mut output = output_storage(len, label)?;
    let reservation = reserve(count, label, None)?;
    let mut local = reservation.first_state;
    for _ in 0..len {
        output.push(next_uniform_state(&mut local));
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
        output.push(next_uniform_state(&mut local));
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
