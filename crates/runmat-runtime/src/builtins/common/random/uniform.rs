//! Contiguous uniform draws from the session's versioned host stream.
//!
//! Reservation is the stream's linearization point. A cancelled or abandoned
//! producer consumes its entire reserved range; it never rewinds the stream.
//! The ordered turn remains held across cooperative yields, while no mutable
//! stream borrow is held across an execution-service turn.

use super::next_uniform_state;
use super::reservation::{checked_length, output_storage, reserve, service_error};
use super::turn::RandomTurn;
use crate::context::RuntimeContext;
use crate::execution::CpuCooperationBudget;
use crate::BuiltinResult;

/// The synchronous caller and cooperative caller use the same reservation and
/// word-to-double conversion. A call's draws remain contiguous while the
/// ordered turn excludes another draw during suspension.
pub(crate) fn generate_uniform(
    turn: &mut RandomTurn,
    len: usize,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    let count = checked_length(len, label)?;
    let mut output = output_storage(len, label)?;
    let reservation = reserve(turn, count, label)?;
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
    turn: &mut RandomTurn,
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
    let reservation = reserve(turn, count, label)?;
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

/// Fill a caller-owned output from one reserved contiguous range. Scaling is
/// performed before the caller's output-class conversion, just as unifrnd's
/// former f64 vector path did, without retaining that vector separately.
pub(crate) async fn fill_uniform_scaled(
    turn: &mut RandomTurn,
    len: usize,
    label: &str,
    context: Option<&RuntimeContext>,
    a: f64,
    b: f64,
    mut write: impl FnMut(usize, f64) -> BuiltinResult<()>,
) -> BuiltinResult<()> {
    let count = checked_length(len, label)?;
    if let Some(context) = context {
        let mut budget =
            CpuCooperationBudget::new(context.execution().clone(), context.cancellation());
        budget
            .checkpoint(0)
            .await
            .map_err(|source| service_error(label, Some(context), source))?;
        let reservation = reserve(turn, count, label)?;
        let mut local = reservation.first_state;
        for index in 0..len {
            write(index, a + (b - a) * next_uniform_state(&mut local))?;
            budget
                .checkpoint(1)
                .await
                .map_err(|source| service_error(label, Some(context), source))?;
        }
        debug_assert_eq!(local, reservation.following_state);
    } else {
        let reservation = reserve(turn, count, label)?;
        let mut local = reservation.first_state;
        for index in 0..len {
            write(index, a + (b - a) * next_uniform_state(&mut local))?;
        }
        debug_assert_eq!(local, reservation.following_state);
    }
    Ok(())
}

#[cfg(test)]
#[path = "uniform/tests.rs"]
mod tests;
