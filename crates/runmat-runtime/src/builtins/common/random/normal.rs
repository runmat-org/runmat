//! Normal draws using the existing Box–Muller pair arithmetic.
//!
//! Every pair consumes two uniform words, including the unused second value
//! of an odd-length result. Reservation is atomic; calculation and shared CPU
//! cooperation occur after releasing the authoritative stream.

use super::next_normal_pair;
use super::reservation::{checked_length, output_storage, reserve, service_error};
use crate::context::RuntimeContext;
use crate::execution::CpuCooperationBudget;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry};

fn draw_count(len: usize, label: &str) -> BuiltinResult<u64> {
    let outputs = checked_length(len, label)?;
    let pairs = outputs / 2 + outputs % 2;
    pairs.checked_mul(2).ok_or_else(|| {
        build_runtime_error(format!(
            "{label}: normal draw count exceeds the stream range"
        ))
        .with_builtin(label)
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .build()
    })
}

fn push_pair(output: &mut Vec<f64>, len: usize, local: &mut u64) {
    let (first, second) = next_normal_pair(local);
    output.push(first);
    if output.len() < len {
        output.push(second);
    }
}

pub(crate) fn generate_normal(len: usize, label: &str) -> BuiltinResult<Vec<f64>> {
    let draws = draw_count(len, label)?;
    let mut output = output_storage(len, label)?;
    let reservation = reserve(draws, label, None)?;
    let mut local = reservation.first_state;
    while output.len() < len {
        push_pair(&mut output, len, &mut local);
    }
    debug_assert_eq!(local, reservation.following_state);
    Ok(output)
}

/// Cooperative equivalent of `generate_normal`. Cancellation or abandonment
/// after reservation consumes the whole range, never a partial prefix.
pub(crate) async fn generate_normal_cooperative(
    len: usize,
    label: &str,
    context: Option<&RuntimeContext>,
    budget: &mut CpuCooperationBudget,
) -> BuiltinResult<Vec<f64>> {
    budget
        .checkpoint(0)
        .await
        .map_err(|source| service_error(label, context, source))?;
    let draws = draw_count(len, label)?;
    let mut output = output_storage(len, label)?;
    budget
        .checkpoint(0)
        .await
        .map_err(|source| service_error(label, context, source))?;
    let reservation = reserve(draws, label, context)?;
    let mut local = reservation.first_state;
    while output.len() < len {
        push_pair(&mut output, len, &mut local);
        budget
            .checkpoint(1)
            .await
            .map_err(|source| service_error(label, context, source))?;
    }
    debug_assert_eq!(local, reservation.following_state);
    Ok(output)
}

#[cfg(test)]
#[path = "normal/tests.rs"]
mod tests;
