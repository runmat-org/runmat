//! Conservative certification of the actually emitted binary64 basis.
use super::Householder;
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};

fn actual(integer: &BigInt) -> Option<BigRational> {
    BigRational::from_float(integer.to_f64().filter(|value| value.is_finite())?)
}

pub(super) async fn admissible(
    model: &Householder,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<bool> {
    if model.norm.bits() <= 53 {
        let mut exact = true;
        if let Some(negative) = &model.negative {
            let mut sum = BigInt::zero();
            for value in &model.q {
                sum += value;
                cooperation.completed(1).await?;
            }
            for (coordinate, value) in model.q.iter().zip(negative) {
                exact &= super::super::generator::exact(value);
                exact &= *value == coordinate * &sum * 2 - &model.norm;
                cooperation.completed(1).await?;
            }
        }
        if exact {
            return Ok(true);
        }
    }
    let mut maximum_error = BigRational::zero();
    let mut maximum_residual = BigRational::zero();
    for row in 0..model.q.len() {
        let mut sum = BigRational::zero();
        for column in 0..model.q.len() {
            let integer = model.coefficient(row, column);
            let Some(rounded) = actual(&integer) else {
                return Ok(false);
            };
            maximum_error =
                maximum_error.max((&rounded - BigRational::from_integer(integer)).abs());
            sum += rounded;
            cooperation.completed(1).await?;
        }
        if let Some(negative) = &model.negative {
            let Some(rounded) = actual(&negative[row]) else {
                return Ok(false);
            };
            maximum_residual = maximum_residual.max((sum + rounded).abs());
        }
    }
    // Exact H has singular values S=||q||². ||E||₂ <= n*max|Eij|.
    // For minimal completion, ||A^-1(A*1+c)||∞ <= n*max|r|/(S-n*max|E|).
    // The strict bound below proves full rank and strictly positive weights;
    // it does not pretend rounded columns remain exactly orthogonal.
    let bound = (maximum_error + maximum_residual) * BigInt::from(model.q.len());
    Ok(BigRational::from_integer(model.norm.clone()) > bound)
}
