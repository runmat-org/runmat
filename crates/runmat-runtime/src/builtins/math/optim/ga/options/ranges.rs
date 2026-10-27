//! Finite sampling intervals are distinct from the original feasible domain.
use super::{Field, Owner, ScalarCooperation};
use crate::BuiltinResult;
use runmat_value::Value;

pub(in super::super) async fn resolve(
    initial: Option<&Value>,
    lower: &[f64],
    upper: &[f64],
    mixed: bool,
    bitstring: bool,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<(f64, f64)>> {
    let n = lower.len();
    if upper.len() != n {
        return Err(Owner::Solver.invalid("bound extent mismatch"));
    }
    let specified = if let Some(value) = initial {
        super::admission::validate_cooperative(
            Owner::Solver,
            Field::InitialPopulationRange,
            value,
            cooperation,
        )
        .await?;
        let Value::Tensor(value) = value else {
            return Err(Owner::Solver.invalid("initial range must be a two-row matrix"));
        };
        if value.shape()[1] != 1 && value.shape()[1] != n {
            return Err(Owner::Solver.invalid("initial range must have one or nvars columns"));
        }
        Some(value)
    } else {
        None
    };
    let mut ranges = super::super::POLICY.capacity(n, cooperation.runtime())?;
    for column in 0..n {
        let (a, b) = if bitstring {
            (0., 1.)
        } else if let Some(value) = specified {
            let column = if value.shape()[1] == 1 { 0 } else { column };
            (
                crate::builtins::common::tensor::tensor_value_f64(value, 2 * column),
                crate::builtins::common::tensor::tensor_value_f64(value, 2 * column + 1),
            )
        } else if mixed {
            (-9999., 10001.)
        } else {
            (-10., 10.)
        };
        ranges.push(if bitstring {
            (a, b)
        } else {
            fit(a, b, lower[column], upper[column], specified.is_some())?
        });
        cooperation.completed(1).await?;
    }
    Ok(ranges)
}
fn fit(a: f64, b: f64, lower: f64, upper: f64, specified: bool) -> BuiltinResult<(f64, f64)> {
    if lower.is_nan()
        || upper.is_nan()
        || lower > upper
        || lower == f64::INFINITY
        || upper == f64::NEG_INFINITY
    {
        return Err(Owner::Solver.invalid("bounds have no finite feasible coordinate"));
    }
    if !specified && lower.is_finite() && upper.is_finite() {
        return Ok((lower, upper));
    }
    if specified && a.max(lower) <= b.min(upper) {
        return Ok((a.max(lower), b.min(upper)));
    }
    let width = b - a;
    let (a, b) = if lower.is_finite() && a < lower {
        (lower, (lower + width).min(upper))
    } else if upper.is_finite() && b > upper {
        ((upper - width).max(lower), upper)
    } else {
        (a, b)
    };
    if !a.is_finite() || !b.is_finite() || a > b {
        return Err(Owner::Solver.invalid("finite initial sampling interval cannot be represented"));
    }
    Ok((a, b))
}
