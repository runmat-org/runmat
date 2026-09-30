//! Native-class numerical steps. Old slots are borrowed; new slots are staged
//! independently so an error cannot partially advance optimizer state.

use crate::builtins::deep_learning::model::execution::host::numeric::Real;
use crate::BuiltinResult;

use super::{controls::NativeControls, fail, slot::SolverSlot};
use crate::builtins::deep_learning::supervised::native::factors::NativeFactors;

fn reserve<T>(length: usize, function: &'static str) -> BuiltinResult<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| fail(function, "cannot reserve optimizer step storage"))?;
    Ok(values)
}

fn source_gradient<T: Real>(
    value: T,
    gradient: T,
    factor: (T, T),
    function: &'static str,
) -> BuiltinResult<(T, T)> {
    let (rate, regularization) = factor;
    let gradient = gradient + regularization * value;
    if !value.is_finite() || !gradient.is_finite() {
        return Err(fail(
            function,
            "optimizer requires finite values and gradients",
        ));
    }
    Ok((rate, gradient))
}

fn finite<T: Real>(values: &[T], function: &'static str) -> BuiltinResult<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(fail(function, "optimizer step exceeds native precision"))
    }
}

pub(super) fn step<T: Real>(
    current: &[T],
    gradient: &[T],
    old: Option<&SolverSlot<T>>,
    controls: NativeControls<T>,
    factors: &NativeFactors<T>,
    function: &'static str,
) -> BuiltinResult<(Vec<T>, SolverSlot<T>)> {
    if current.len() != gradient.len() {
        return Err(fail(function, "optimizer gradient geometry changed"));
    }
    let mut next = reserve(current.len(), function)?;
    let slot = match controls {
        NativeControls::Sgdm {
            momentum_complement,
        } => {
            let previous = match old {
                Some(SolverSlot::Sgdm { velocity }) if velocity.len() == current.len() => {
                    Some(velocity.as_slice())
                }
                None => None,
                _ => return Err(fail(function, "SGDM slot geometry or solver changed")),
            };
            let mut velocity = reserve(current.len(), function)?;
            for index in 0..current.len() {
                let (rate, grad) =
                    source_gradient(current[index], gradient[index], factors.at(index), function)?;
                let old_velocity = previous.map_or(T::zero(), |values| values[index]);
                let movement =
                    old_velocity + momentum_complement * (T::zero() - old_velocity) - rate * grad;
                velocity.push(movement);
                next.push(current[index] + movement);
            }
            finite(&velocity, function)?;
            SolverSlot::Sgdm { velocity }
        }
        NativeControls::Rmsprop {
            squared_complement,
            epsilon,
        } => {
            let previous = match old {
                Some(SolverSlot::Rmsprop { squared }) if squared.len() == current.len() => {
                    Some(squared.as_slice())
                }
                None => None,
                _ => return Err(fail(function, "RMSProp slot geometry or solver changed")),
            };
            let mut squared = reserve(current.len(), function)?;
            for index in 0..current.len() {
                let (rate, grad) =
                    source_gradient(current[index], gradient[index], factors.at(index), function)?;
                let old_squared = previous.map_or(T::zero(), |values| values[index]);
                let signal = grad * grad;
                let moment = old_squared + squared_complement * (signal - old_squared);
                squared.push(moment);
                next.push(current[index] - rate * grad / (moment.sqrt() + epsilon));
            }
            finite(&squared, function)?;
            SolverSlot::Rmsprop { squared }
        }
        NativeControls::Adam {
            mean_complement,
            squared_complement,
            mean_correction,
            squared_correction,
            epsilon,
        } => {
            let previous = match old {
                Some(SolverSlot::Adam { mean, squared })
                    if mean.len() == current.len() && squared.len() == current.len() =>
                {
                    Some((mean.as_slice(), squared.as_slice()))
                }
                None => None,
                _ => return Err(fail(function, "Adam slot geometry or solver changed")),
            };
            let mut mean = reserve(current.len(), function)?;
            let mut squared = reserve(current.len(), function)?;
            for index in 0..current.len() {
                let (rate, grad) =
                    source_gradient(current[index], gradient[index], factors.at(index), function)?;
                let old_mean = previous.map_or(T::zero(), |values| values.0[index]);
                let old_squared = previous.map_or(T::zero(), |values| values.1[index]);
                let new_mean = old_mean + mean_complement * (grad - old_mean);
                let signal = grad * grad;
                let new_squared = old_squared + squared_complement * (signal - old_squared);
                mean.push(new_mean);
                squared.push(new_squared);
                next.push(
                    current[index]
                        - rate * (new_mean / mean_correction)
                            / ((new_squared / squared_correction).sqrt() + epsilon),
                );
            }
            finite(&mean, function)?;
            finite(&squared, function)?;
            SolverSlot::Adam { mean, squared }
        }
    };
    finite(&next, function)?;
    Ok((next, slot))
}
