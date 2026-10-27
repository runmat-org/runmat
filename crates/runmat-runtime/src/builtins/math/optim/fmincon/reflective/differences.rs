//! Bound-interior gradient differences with one retained accepted gradient,
//! shared structural coloring and a complete sweep/trial budget reservation.
use super::super::{
    derivatives::Difference,
    evaluation::{EvaluationStop, Point},
};
use super::*;
use crate::builtins::math::optim::common::derivative_scaling::DerivativeScaling;
use crate::execution::RootedValue;
use runmat_value::{SparseTensor, Tensor, Value};
pub(super) use trust_quadratic::pattern::Pattern;

pub(super) enum Stop {
    Time,
    Budget,
    Undefined,
}
fn quotient(value: f64, base: f64, offset: f64) -> Option<f64> {
    let difference = value - base;
    let result = if difference.is_finite() {
        difference / offset
    } else {
        value / offset - base / offset
    };
    result.is_finite().then_some(result)
}
pub(super) async fn build(
    x: &[f64],
    point: &Point,
    geometry: &Geometry,
    pattern: Option<&Pattern>,
    mode: Difference,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<std::result::Result<model::Model, Stop>> {
    let n = x.len();
    let colors = pattern.map_or(n, |p| p.count);
    // TR-reflective does not consume TypicalX. This immutable default uses
    // the same canonical representable-point rules, not dormant user data.
    let scaling = DerivativeScaling::default();
    let mut points = workspace::capacity(n, IDENTITY)?;
    for (index, &x) in x.iter().enumerate() {
        if pattern.is_some_and(|pattern| pattern.colors[index] == usize::MAX) {
            points.push(None);
            evaluator.cooperation.completed(1).await?;
            continue;
        }
        let lower = if geometry.lower[index].is_finite() {
            geometry.lower[index].next_up()
        } else {
            f64::NEG_INFINITY
        };
        let upper = if geometry.upper[index].is_finite() {
            geometry.upper[index].next_down()
        } else {
            f64::INFINITY
        };
        let Some(probe) = scaling.points(
            "fmincon",
            x,
            index,
            lower,
            upper,
            mode == Difference::Central,
        )?
        else {
            return Ok(Err(Stop::Undefined));
        };
        points.push(Some(probe));
        evaluator.cooperation.completed(1).await?;
    }
    let mut cost = 0usize;
    for color in 0..colors {
        let mut second = false;
        for (column, probe) in points.iter().enumerate() {
            if pattern.map_or(column, |p| p.colors[column]) == color {
                second |= probe
                    .as_ref()
                    .expect("active Hessian color")
                    .second
                    .is_some();
            }
            evaluator.cooperation.completed(1).await?;
        }
        cost = cost
            .checked_add(if second { 2 } else { 1 })
            .ok_or_else(|| super::super::invalid("Hessian point budget overflow"))?;
    }
    if !evaluator.budget.permits_sweep_and_trial(cost) {
        return Ok(Err(Stop::Budget));
    }
    let size = match pattern {
        Some(p) => p.rows.len(),
        None => n
            .checked_mul(n)
            .ok_or_else(|| super::super::invalid("Hessian extent overflow"))?,
    };
    let mut values = zeros(size, evaluator.cooperation).await?;
    let mut probe = copy(x, evaluator.cooperation).await?;
    let base = point
        .gradient
        .as_ref()
        .expect("required supplied objective gradient");
    for color in 0..colors {
        for (column, probe_point) in points.iter().enumerate() {
            if pattern.map_or(column, |p| p.colors[column]) == color {
                probe[column] = probe_point.as_ref().expect("active Hessian color").first;
            }
            evaluator.cooperation.completed(1).await?;
        }
        let first = match evaluator.evaluate_reflective(&probe, false).await? {
            Ok((point, None)) => point,
            Ok((_, Some(_))) => unreachable!("two requested outputs have no third root"),
            Err(EvaluationStop::Time) => return Ok(Err(Stop::Time)),
            Err(EvaluationStop::BudgetExhausted) => return Ok(Err(Stop::Budget)),
            Err(EvaluationStop::Undefined) => return Ok(Err(Stop::Undefined)),
        };
        let mut needs_second = false;
        for (column, probe_point) in points.iter().enumerate() {
            if pattern.map_or(column, |p| p.colors[column]) == color {
                let probe_point = probe_point.as_ref().expect("active Hessian color");
                probe[column] = probe_point.second.unwrap_or(x[column]);
                needs_second |= probe_point.second.is_some();
            }
            evaluator.cooperation.completed(1).await?;
        }
        let second = if needs_second {
            match evaluator.evaluate_reflective(&probe, false).await? {
                Ok((point, None)) => Some(point),
                Ok((_, Some(_))) => unreachable!("two requested outputs have no third root"),
                Err(EvaluationStop::Time) => return Ok(Err(Stop::Time)),
                Err(EvaluationStop::BudgetExhausted) => return Ok(Err(Stop::Budget)),
                Err(EvaluationStop::Undefined) => return Ok(Err(Stop::Undefined)),
            }
        } else {
            None
        };
        let first_gradient = first.gradient.as_ref().expect("supplied probe gradient");
        for (column, probe_point) in points.iter().enumerate() {
            evaluator.cooperation.completed(1).await?;
            if pattern.map_or(column, |p| p.colors[column]) != color {
                continue;
            }
            let probe_point = probe_point.as_ref().expect("active Hessian color");
            let begin = pattern.map_or_else(|| column * n, |p| p.columns[column]);
            let end = pattern.map_or_else(|| (column + 1) * n, |p| p.columns[column + 1]);
            let h = probe_point.first - x[column];
            for (index, value) in values.iter_mut().enumerate().take(end).skip(begin) {
                let row = pattern.map_or(index - begin, |p| p.rows[index]);
                let Some(first) = quotient(first_gradient[row], base[row], h) else {
                    return Ok(Err(Stop::Undefined));
                };
                *value = if let Some(second_x) = probe_point.second {
                    let k = second_x - x[column];
                    let second_gradient = second
                        .as_ref()
                        .expect("color has a central probe")
                        .gradient
                        .as_ref()
                        .expect("supplied second gradient");
                    let Some(second) = quotient(second_gradient[row], base[row], k) else {
                        return Ok(Err(Stop::Undefined));
                    };
                    (-k / (h - k)) * first + (h / (h - k)) * second
                } else {
                    first
                };
                if !value.is_finite() {
                    return Ok(Err(Stop::Undefined));
                }
                evaluator.cooperation.completed(1).await?;
            }
            probe[column] = x[column];
        }
    }
    if let Some(pattern) = pattern {
        for column in 0..n {
            for index in pattern.columns[column]..pattern.columns[column + 1] {
                let row = pattern.rows[index];
                if row <= column {
                    continue;
                }
                let other = pattern.columns[row]
                    + pattern.rows[pattern.columns[row]..pattern.columns[row + 1]]
                        .binary_search(&column)
                        .expect("symmetric shared pattern");
                let value = 0.5 * values[index] + 0.5 * values[other];
                values[index] = value;
                values[other] = value;
                evaluator.cooperation.completed(1).await?;
            }
        }
    } else {
        for column in 0..n {
            for row in column + 1..n {
                let value = 0.5 * values[column * n + row] + 0.5 * values[row * n + column];
                values[column * n + row] = value;
                values[row * n + column] = value;
                evaluator.cooperation.completed(1).await?;
            }
        }
    }
    let value = match pattern {
        Some(pattern) => Value::SparseTensor(
            SparseTensor::new(n, n, pattern.columns.clone(), pattern.rows.clone(), values)
                .map_err(super::super::invalid)?,
        ),
        None => Value::Tensor(Tensor::new(values, vec![n, n]).map_err(super::super::invalid)?),
    };
    let value = RootedValue::in_optional_context(value, evaluator.cooperation.runtime())?;
    model::matrix(value, n, evaluator.cooperation).await.map(Ok)
}
