//! A rare ambiguous binary64 completion is checked in its actual basis.
use super::super::{buffer, invalid};
use super::generator::Generator;
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};
fn actual(value: &BigInt) -> Option<BigRational> {
    BigRational::from_float(value.to_f64().filter(|value| value.is_finite())?)
}
pub(super) async fn positive(
    model: &Generator,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    let negative = model
        .negative
        .as_ref()
        .ok_or_else(|| invalid("missing MADS minimal completion"))?;
    let pivot = model.seed.pivot;
    let (Some(target), Some(diagonal)) =
        (actual(&negative[pivot]), actual(&model.seed.values[pivot]))
    else {
        return Ok(false);
    };
    if diagonal.is_zero() {
        return Ok(false);
    }
    let seed_weight = -target / diagonal;
    if !seed_weight.is_positive() {
        return Ok(false);
    }
    let mut weights = buffer(model.dimension - 1)?;
    for row in 0..model.dimension - 1 {
        let physical = model.rows[row];
        let (Some(target), Some(seed)) = (
            actual(&negative[physical]),
            actual(&model.seed.values[physical]),
        ) else {
            return Ok(false);
        };
        let mut residual = -target - seed * &seed_weight;
        for (column, weight) in weights.iter().enumerate() {
            let Some(coefficient) = actual(&model.coefficients[model.starts[row] + column]) else {
                return Ok(false);
            };
            residual -= coefficient * weight;
            cooperation.completed(1).await?;
        }
        let Some(diagonal) = actual(&model.coefficients[model.starts[row] + row]) else {
            return Ok(false);
        };
        if diagonal.is_zero() {
            return Ok(false);
        }
        let weight = residual / diagonal;
        if !weight.is_positive() {
            return Ok(false);
        }
        weights.push(weight);
        cooperation.completed(1).await?;
    }
    Ok(true)
}
