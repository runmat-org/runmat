//! Host normal transforms for versioned RNG streams.

mod ziggurat;

use super::reservation::{checked_length, output_storage, service_error};
use super::turn::RandomTurn;
use super::RandomState;
use super::{next_uniform_state, MIN_UNIFORM};
use crate::context::RuntimeContext;
use crate::execution::CpuCooperationBudget;
use crate::BuiltinResult;
use std::f64::consts::PI;

/// The shipped LCG wire keeps its historical two-uniform Box–Muller pairing.
/// MT uses two independent Ziggurat draws, each consuming a variable number
/// of MT words. This is one numerical owner for randn and distribution helpers.
pub(super) fn next_normal_pair(state: &mut RandomState) -> (f64, f64) {
    match state {
        RandomState::Lcg(_) => {
            let u1 = next_uniform_state(state).max(MIN_UNIFORM);
            let u2 = next_uniform_state(state);
            let radius = (-2.0 * u1.ln()).sqrt();
            let angle = 2.0 * PI * u2;
            (radius * angle.cos(), radius * angle.sin())
        }
        RandomState::Mt(_) => (ziggurat::next(state), ziggurat::next(state)),
        RandomState::MtSeed(_) => panic!("normal draw requires admitted MT materialization"),
    }
}

pub(super) fn next_normal(state: &mut RandomState) -> f64 {
    match state {
        RandomState::Mt(_) => ziggurat::next(state),
        RandomState::Lcg(_) => next_normal_pair(state).0,
        RandomState::MtSeed(_) => panic!("normal draw requires admitted MT materialization"),
    }
}

fn push_pair(output: &mut Vec<f64>, len: usize, local: &mut RandomState) {
    match local {
        RandomState::Mt(_) => output.push(next_normal(local)),
        RandomState::Lcg(_) => {
            let (first, second) = next_normal_pair(local);
            output.push(first);
            if output.len() < len {
                output.push(second);
            }
        }
        RandomState::MtSeed(_) => panic!("normal draw requires admitted MT materialization"),
    }
}

fn fill_next_batch(
    state: &mut RandomState,
    index: &mut usize,
    len: usize,
    write: &mut impl FnMut(usize, f64) -> BuiltinResult<()>,
) -> BuiltinResult<()> {
    if matches!(state, RandomState::Lcg(_)) {
        let (first, second) = next_normal_pair(state);
        write(*index, first)?;
        *index += 1;
        if *index < len {
            write(*index, second)?;
            *index += 1;
        }
    } else {
        write(*index, next_normal(state))?;
        *index += 1;
    }
    Ok(())
}

pub(crate) fn fill_normal(
    turn: &mut RandomTurn,
    len: usize,
    label: &str,
    mut write: impl FnMut(usize, f64) -> BuiltinResult<()>,
) -> BuiltinResult<()> {
    checked_length(len, label)?;
    if len == 0 {
        return Ok(());
    }
    let owner = turn.owner().cloned();
    turn.with_state(label, |rng| -> BuiltinResult<()> {
        rng.state.prepare_mutation(owner.as_ref())?;
        let mut index = 0;
        while index < len {
            fill_next_batch(&mut rng.state, &mut index, len, &mut write)?;
        }
        Ok(())
    })?
}

/// Fill an already admitted output without constructing a second numeric
/// payload. One LCG pair remains one draw batch, including an unused odd tail;
/// MT emits one independently sampled value per batch.
pub(crate) async fn fill_normal_cooperative(
    turn: &mut RandomTurn,
    len: usize,
    label: &str,
    context: &RuntimeContext,
    budget: &mut CpuCooperationBudget,
    mut write: impl FnMut(usize, f64) -> BuiltinResult<()>,
) -> BuiltinResult<()> {
    budget
        .checkpoint(0)
        .await
        .map_err(|source| service_error(label, Some(context), source))?;
    checked_length(len, label)?;
    let owner = turn.owner().cloned();
    let mut index = 0;
    while index < len {
        turn.with_state(label, |rng| -> BuiltinResult<()> {
            rng.state.prepare_mutation(owner.as_ref())?;
            fill_next_batch(&mut rng.state, &mut index, len, &mut write)
        })??;
        budget
            .checkpoint(1)
            .await
            .map_err(|source| service_error(label, Some(context), source))?;
    }
    Ok(())
}

pub(crate) fn generate_normal(
    turn: &mut RandomTurn,
    len: usize,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    checked_length(len, label)?;
    let mut output = output_storage(len, label)?;
    if len == 0 {
        return Ok(output);
    }
    let owner = turn.owner().cloned();
    turn.with_state(label, |rng| -> BuiltinResult<()> {
        rng.state.prepare_mutation(owner.as_ref())?;
        while output.len() < len {
            push_pair(&mut output, len, &mut rng.state);
        }
        Ok(())
    })??;
    Ok(output)
}

/// Cooperative equivalent. The ordered turn persists across yields, but the
/// stream borrow does not. Cancellation commits exactly the consumed prefix.
pub(crate) async fn generate_normal_cooperative(
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
    checked_length(len, label)?;
    let mut output = output_storage(len, label)?;
    budget
        .checkpoint(0)
        .await
        .map_err(|source| service_error(label, context, source))?;
    if len == 0 {
        return Ok(output);
    }
    let owner = turn.owner().cloned();
    while output.len() < len {
        turn.with_state(label, |rng| -> BuiltinResult<()> {
            rng.state.prepare_mutation(owner.as_ref())?;
            push_pair(&mut output, len, &mut rng.state);
            Ok(())
        })??;
        budget
            .checkpoint(1)
            .await
            .map_err(|source| service_error(label, context, source))?;
    }
    Ok(output)
}

#[cfg(test)]
#[path = "normal/tests.rs"]
mod tests;
