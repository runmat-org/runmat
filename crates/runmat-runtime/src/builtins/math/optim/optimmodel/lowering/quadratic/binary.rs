//! Canonical broadcast/matrix geometry, with exact sparse polynomial rows.
use super::*;
use crate::builtins::common::broadcast::BroadcastPlan;
use runmat_types::OperatorKind as Op;
mod sparse;
pub(super) async fn binary(
    left: Array,
    right: Array,
    operator: Op,
    shape: SharedShape,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> Lowered<Array> {
    if (operator == Op::Mrdivide && right.count(policy)? != 1)
        || (operator == Op::Mldivide && left.count(policy)? != 1)
    {
        return Ok(Err(Stop::NonAffine));
    }
    if operator == Op::MatrixMultiply && left.count(policy)? != 1 && right.count(policy)? != 1 {
        return matrix(left, right, shape, policy, c).await;
    }
    if operator == Op::MatrixPower && left.count(policy)? != 1 {
        if right.count(policy)? != 1 {
            return Ok(Err(Stop::NonAffine));
        }
        let Some(exponent) = right.at(0, policy)?.and_then(|value| value.constant()) else {
            return Ok(Err(Stop::NonAffine));
        };
        if exponent.exact.is_zero() {
            let identity = super::super::powers::identity(shape.clone(), policy, c).await?;
            return Ok(Ok(Array::constant(shape, identity.offset)));
        }
        if exponent
            .exact
            .compare(&Rational::from_usize(1), policy, c)
            .await?
            .is_eq()
        {
            return Ok(Ok(left.reshaped(shape, policy)?));
        }
        if exponent.proposal != 2.
            || exponent
                .exact
                .compare(&Rational::from_usize(2), policy, c)
                .await?
                .is_ne()
        {
            return Ok(Err(Stop::NonAffine));
        }
        return matrix(left.clone(), left, shape, policy, c).await;
    }
    let plan = BroadcastPlan::new(&left.shape, &right.shape).map_err(|e| policy.invalid(e))?;
    if !crate::builtins::common::shape::array_shapes_equal(plan.output_shape(), &shape) {
        return Err(policy.invalid("quadratic arithmetic differs from original broadcast geometry"));
    }
    let mut rows = runmat_value::exact_capacity_vec(plan.len()).map_err(|e| policy.storage(e))?;
    for (_, a, b) in plan.iter() {
        let value = match (left.at(a, policy)?, right.at(b, policy)?) {
            (Some(a), Some(b)) => match operator {
                Op::Add | Op::Subtract => {
                    Some(a.add(b, operator == Op::Subtract, policy, c).await?)
                }
                Op::MatrixMultiply | Op::ElementwiseMultiply => {
                    match a.product(&b, policy, c).await? {
                        Ok(value) => Some(value),
                        Err(stop) => return Ok(Err(stop)),
                    }
                }
                Op::Mrdivide | Op::ElementwiseDivide | Op::Mldivide | Op::ElementwiseLeftDivide => {
                    let (numerator, divisor) =
                        if matches!(operator, Op::Mldivide | Op::ElementwiseLeftDivide) {
                            (&b, &a)
                        } else {
                            (&a, &b)
                        };
                    let Some(divisor) = divisor.constant() else {
                        return Ok(Err(Stop::NonAffine));
                    };
                    if divisor.exact.is_zero() {
                        return Ok(Err(Stop::NonFinite));
                    }
                    let factor = Weight::constant(1., policy)?
                        .divided(&divisor, policy, c)
                        .await?;
                    Some(numerator.scaled(&factor, policy, c).await?)
                }
                Op::MatrixPower | Op::ElementwisePower => {
                    let Some(exponent) = b.constant() else {
                        return Ok(Err(Stop::NonAffine));
                    };
                    if exponent.exact.is_zero() {
                        Some(Polynomial::scalar(1., policy)?)
                    } else if exponent
                        .exact
                        .compare(&Rational::from_usize(1), policy, c)
                        .await?
                        .is_eq()
                    {
                        Some(a)
                    } else if exponent
                        .exact
                        .compare(&Rational::from_usize(2), policy, c)
                        .await?
                        .is_eq()
                    {
                        match a.product(&a, policy, c).await? {
                            Ok(value) => Some(value),
                            Err(stop) => return Ok(Err(stop)),
                        }
                    } else {
                        return Ok(Err(Stop::NonAffine));
                    }
                }
                _ => return Ok(Err(Stop::NonAffine)),
            },
            _ => None,
        };
        rows.push(value);
        c.completed(1).await?;
    }
    Ok(Ok(Array::rows(shape, rows, policy)?))
}
async fn matrix(
    left: Array,
    right: Array,
    shape: SharedShape,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> Lowered<Array> {
    if left.shape.iter().skip(2).any(|&extent| extent != 1)
        || right.shape.iter().skip(2).any(|&extent| extent != 1)
    {
        return Ok(Err(Stop::NonAffine));
    }
    let m = left.shape[0];
    let k = left.shape[1];
    let n = right.shape[1];
    if right.shape[0] != k || !crate::builtins::common::shape::array_shapes_equal(&[m, n], &shape) {
        return Err(policy.invalid("quadratic matrix product differs from original geometry"));
    }
    if let Some(matrix) = right.sparse_constant() {
        if left.all_assigned(policy, c).await? {
            return Ok(Ok(
                sparse::right(&left, matrix, shape, m, n, policy, c).await?
            ));
        }
    }
    if let Some(matrix) = left.sparse_constant() {
        if right.all_assigned(policy, c).await? {
            return Ok(Ok(
                sparse::left(matrix, &right, shape, m, n, policy, c).await?
            ));
        }
    }
    let count = m
        .checked_mul(n)
        .ok_or_else(|| policy.invalid("quadratic matrix product overflows usize"))?;
    let mut rows = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
    for column in 0..n {
        for row in 0..m {
            let mut value = Some(Polynomial::zero());
            for axis in 0..k {
                value = match (
                    value,
                    left.at(row + axis * m, policy)?,
                    right.at(axis + column * k, policy)?,
                ) {
                    (Some(sum), Some(a), Some(b)) => match a.product(&b, policy, c).await? {
                        Ok(product) => Some(sum.add(product, false, policy, c).await?),
                        Err(stop) => return Ok(Err(stop)),
                    },
                    _ => None,
                };
                c.completed(1).await?;
            }
            rows.push(value);
            c.completed(1).await?;
        }
    }
    Ok(Ok(Array::rows(shape, rows, policy)?))
}
