//! Hessian differences of supplied gradients. A structural pattern is
//! lowered once and graph-colored; one gradient call serves each color.
//! The accepted objective and gradient are never replayed for this model.
use super::super::evaluation::{ObjectiveEvaluator, ObjectiveState};
use super::{model::Model, *};
use runmat_value::SparseTensor;

pub(super) use crate::builtins::math::optim::common::trust_quadratic::pattern::Pattern;

pub(super) async fn build(
    state: &ObjectiveState,
    pattern: Option<&Pattern>,
    evaluator: &mut ObjectiveEvaluator<'_>,
) -> BuiltinResult<Option<Model>> {
    let Some(value) = build_value(state, pattern, evaluator, 1).await? else {
        return Ok(None);
    };
    Model::matrix(value, state.x.len(), evaluator.cooperation)
        .await
        .map(Some)
}

pub(in crate::builtins::math::optim::fminunc) async fn requested_hessian(
    state: &ObjectiveState,
    evaluator: &mut ObjectiveEvaluator<'_>,
) -> BuiltinResult<Option<crate::execution::RootedValue>> {
    build_value(state, None, evaluator, 0).await
}

async fn build_value(
    state: &ObjectiveState,
    pattern: Option<&Pattern>,
    evaluator: &mut ObjectiveEvaluator<'_>,
    reserve_trial: usize,
) -> BuiltinResult<Option<crate::execution::RootedValue>> {
    let n = state.x.len();
    let count = pattern.map_or(n, |pattern| pattern.count);
    // Reserve the entire model's callbacks and one trial before allocating
    // or invoking a partial difference sweep. Exhaustion is a truthful limit.
    if count
        .checked_mul(
            evaluator
                .derivatives
                .state_cost(n)
                .ok_or_else(|| invalid("Hessian evaluation cost overflow"))?,
        )
        .and_then(|cost| cost.checked_add(reserve_trial))
        .is_none_or(|needed| needed > evaluator.remaining_func_evals())
    {
        return Ok(None);
    }
    let length = match pattern {
        Some(pattern) => pattern.rows.len(),
        None => n
            .checked_mul(n)
            .ok_or_else(|| invalid("Hessian extent overflow"))?,
    };
    let mut values = zeros(length, evaluator.cooperation).await?;
    let mut x = buffer(n)?;
    x.extend_from_slice(&state.x);
    let mut deltas = zeros(n, evaluator.cooperation).await?;
    for column in 0..n {
        let relative = if evaluator.derivatives
            == super::super::algorithm::finite_difference::DerivativeMode::Analytic
        {
            f64::EPSILON.sqrt()
        } else {
            f64::EPSILON.sqrt().sqrt()
        };
        let nominal = relative * state.x[column].abs().max(1.0);
        let point = if (state.x[column] + nominal).is_finite() {
            state.x[column] + nominal
        } else {
            state.x[column] - nominal
        };
        deltas[column] = point - state.x[column];
        if deltas[column] == 0.0 || !deltas[column].is_finite() {
            return Err(invalid(
                "cannot represent a Hessian difference displacement",
            ));
        }
        evaluator.cooperation.completed(1).await?;
    }
    for color in 0..count {
        for column in 0..n {
            evaluator.cooperation.completed(1).await?;
            if pattern.map_or(column, |pattern| pattern.colors[column]) == color {
                x[column] = state.x[column] + deltas[column];
            }
        }
        let perturbed = evaluator.evaluate(&x).await?;
        for column in 0..n {
            if pattern.map_or(column, |pattern| pattern.colors[column]) != color {
                continue;
            }
            let first = pattern.map_or_else(|| column * n, |pattern| pattern.columns[column]);
            let end =
                pattern.map_or_else(|| (column + 1) * n, |pattern| pattern.columns[column + 1]);
            for index in first..end {
                let row = pattern.map_or_else(|| index - column * n, |pattern| pattern.rows[index]);
                let difference = perturbed.grad[row] - state.grad[row];
                let value = if difference.is_finite() {
                    difference / deltas[column]
                } else {
                    perturbed.grad[row] / deltas[column] - state.grad[row] / deltas[column]
                };
                if !value.is_finite() {
                    return Err(invalid("gradient differences produced a nonfinite Hessian"));
                }
                values[index] = value;
                evaluator.cooperation.completed(1).await?;
            }
            x[column] = state.x[column];
        }
    }
    // Differences are numerically asymmetric; the quadratic model uses
    // the symmetric part, not a triangle selected by storage accident.
    if let Some(pattern) = pattern {
        for column in 0..n {
            for index in pattern.columns[column]..pattern.columns[column + 1] {
                let row = pattern.rows[index];
                if row <= column {
                    continue;
                }
                let offset = pattern.rows[pattern.columns[row]..pattern.columns[row + 1]]
                    .binary_search(&column)
                    .expect("validated symmetric structure");
                let other = pattern.columns[row] + offset;
                let average = values[index] * 0.5 + values[other] * 0.5;
                values[index] = average;
                values[other] = average;
                evaluator.cooperation.completed(1).await?;
            }
        }
    } else {
        for column in 0..n {
            for row in column + 1..n {
                let average = values[column * n + row] * 0.5 + values[row * n + column] * 0.5;
                values[column * n + row] = average;
                values[row * n + column] = average;
                evaluator.cooperation.completed(1).await?;
            }
        }
    }
    let value = if let Some(pattern) = pattern {
        SparseTensor::new(n, n, pattern.columns.clone(), pattern.rows.clone(), values)
            .map(Value::SparseTensor)
            .map_err(invalid)?
    } else {
        Tensor::new(values, vec![n, n])
            .map(Value::Tensor)
            .map_err(invalid)?
    };
    let value =
        crate::execution::RootedValue::in_optional_context(value, evaluator.cooperation.runtime())?;
    Ok(Some(value))
}
