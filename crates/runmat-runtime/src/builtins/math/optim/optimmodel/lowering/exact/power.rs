//! Original integer powers. Floating exponent proposals were already checked
//! against the exact immutable exponent DAG before compiler admission.
use super::super::powers::Exponents;
use super::*;
use runmat_types::OperatorKind as Op;
mod inverse;
pub(super) async fn evaluate(
    left: Data,
    right: Data,
    operator: Op,
    exponents: &Exponents,
    shape: SharedShape,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Data> {
    if operator == Op::MatrixPower && left.count(policy)? != 1 {
        let mut exponent = exponents.at(0, policy)?;
        if exponent == 1. {
            return left.reshaped(shape, policy);
        }
        if exponent == 0. {
            let identity = super::super::powers::identity(shape.clone(), policy, c).await?;
            return Ok(Data::constant(shape, identity.offset));
        }
        if left.diagonal(policy, c).await? {
            let n = left.shape.first().copied().unwrap_or(1);
            let mut values = HashMap::new();
            for axis in 0..n {
                let row = axis + axis * n;
                let Some(value) = left.at(row, policy)? else {
                    return Ok(Data::uniform(shape, None));
                };
                let Some(value) = scalar(value, exponent, policy, c).await? else {
                    return Ok(Data::uniform(shape, None));
                };
                if !value.is_zero() {
                    values.try_reserve(1).map_err(|e| policy.reserve(e))?;
                    values.insert(row, value);
                }
                c.completed(1).await?;
            }
            return Ok(Data::routed(shape, values));
        }
        let mut base = if exponent < 0. {
            exponent = -exponent;
            inverse::evaluate(left, policy, c).await?
        } else {
            left
        };
        let identity = super::super::powers::identity(shape.clone(), policy, c).await?;
        let mut result = Data::constant(shape.clone(), identity.offset);
        while exponent >= 1. {
            if exponent % 2. == 1. {
                result = arithmetic::binary(
                    result,
                    base.clone(),
                    Op::MatrixMultiply,
                    shape.clone(),
                    true,
                    policy,
                    c,
                )
                .await?;
            }
            exponent = (exponent / 2.).floor();
            if exponent >= 1. {
                base = arithmetic::binary(
                    base.clone(),
                    base,
                    Op::MatrixMultiply,
                    shape.clone(),
                    true,
                    policy,
                    c,
                )
                .await?;
            }
            c.completed(1).await?;
        }
        return Ok(result);
    }
    let plan = crate::builtins::common::broadcast::BroadcastPlan::new(&left.shape, &right.shape)
        .map_err(|e| policy.invalid(e))?;
    if !crate::builtins::common::shape::array_shapes_equal(plan.output_shape(), &shape) {
        return Err(policy.invalid("exact power differs from original broadcast"));
    }
    if exponents.count(policy)? == 1 {
        if let Some(value) = left.uniform_value() {
            let value = match value {
                Some(value) => scalar(value.clone(), exponents.at(0, policy)?, policy, c).await?,
                None => None,
            };
            return Ok(Data::uniform(shape, value));
        }
    }
    let mut values = data::buffer(plan.len(), policy)?;
    for (row, a, _) in plan.iter() {
        values.push(match left.at(a, policy)? {
            Some(value) => {
                scalar(value, exponents.broadcast(row, &shape, policy)?, policy, c).await?
            }
            None => None,
        });
        c.completed(1).await?;
    }
    Data::explicit(shape, values, policy)
}
async fn scalar(
    mut base: Rational,
    mut exponent: f64,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Option<Rational>> {
    let one = Rational::from_usize(1);
    if exponent < 0. {
        if base.is_zero() {
            return Ok(None);
        }
        base = one.div(&base, policy, c).await?;
        exponent = -exponent;
    }
    let mut value = one;
    while exponent >= 1. {
        if exponent % 2. == 1. {
            value = value.mul(&base, policy, c).await?;
        }
        exponent = (exponent / 2.).floor();
        if exponent >= 1. {
            base = base.mul(&base, policy, c).await?;
        }
        c.completed(1).await?;
    }
    Ok(Some(value))
}
