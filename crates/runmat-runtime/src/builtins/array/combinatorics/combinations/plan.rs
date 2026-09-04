use runmat_builtins::COMBINATIONS_ERROR_TOO_LARGE;

use crate::BuiltinResult;

use super::error;

const MAX_ROWS: usize = 50_000_000;

#[derive(Debug, Clone, Copy)]
pub(super) struct Repetition {
    pub outer: usize,
    pub inner: usize,
}

#[derive(Debug)]
pub(super) struct CartesianPlan {
    pub rows: usize,
    pub repetitions: Vec<Repetition>,
}

pub(super) fn build(lengths: &[usize]) -> BuiltinResult<CartesianPlan> {
    let rows = checked_product(lengths)?;
    if rows > MAX_ROWS {
        return Err(too_large("combinations: output is too large"));
    }
    let repetitions = (0..lengths.len())
        .map(|column| {
            Ok(Repetition {
                outer: checked_product(&lengths[..column])?,
                inner: checked_product(&lengths[column + 1..])?,
            })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    Ok(CartesianPlan { rows, repetitions })
}

fn checked_product(lengths: &[usize]) -> BuiltinResult<usize> {
    lengths.iter().try_fold(1_usize, |product, length| {
        product
            .checked_mul(*length)
            .ok_or_else(|| too_large("combinations: output row count overflow"))
    })
}

fn too_large(message: &'static str) -> crate::RuntimeError {
    error::with_message(&COMBINATIONS_ERROR_TOO_LARGE, message)
}
