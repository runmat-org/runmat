//! Exact source arithmetic at one candidate, not a replacement numerical
//! optimizer. Ordinary floating evaluation retains its separate contract.
use super::*;
use runmat_types::OperatorKind as Op;
mod sparse;
async fn scalar(
    operator: Op,
    a: Rational,
    b: Rational,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Rational> {
    match operator {
        Op::Add => a.add(&b, policy, cooperation).await,
        Op::Subtract => a.sub(&b, policy, cooperation).await,
        Op::ElementwiseMultiply | Op::MatrixMultiply => a.mul(&b, policy, cooperation).await,
        Op::ElementwiseDivide | Op::Mrdivide => {
            if b.is_zero() {
                Err(policy.invalid("exact affine divisor is zero"))
            } else {
                a.div(&b, policy, cooperation).await
            }
        }
        Op::ElementwiseLeftDivide | Op::Mldivide => {
            if a.is_zero() {
                Err(policy.invalid("exact affine divisor is zero"))
            } else {
                b.div(&a, policy, cooperation).await
            }
        }
        _ => Err(policy.invalid("unadmitted exact arithmetic operation")),
    }
}
pub(super) async fn binary(
    left: Data,
    right: Data,
    operator: Op,
    shape: SharedShape,
    complete: bool,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Data> {
    if operator == Op::MatrixMultiply && left.count(policy)? != 1 && right.count(policy)? != 1 {
        return matrix(left, right, shape, complete, policy, cooperation).await;
    }
    let plan = crate::builtins::common::broadcast::BroadcastPlan::new(&left.shape, &right.shape)
        .map_err(|e| policy.invalid(e))?;
    if !crate::builtins::common::shape::array_shapes_equal(plan.output_shape(), &shape) {
        return Err(policy.invalid("exact broadcast differs from original expression geometry"));
    }

    if right
        .uniform_value()
        .is_some_and(|value| value.as_ref().is_some_and(Rational::is_zero))
        && matches!(operator, Op::Add | Op::Subtract)
        && left.count(policy)? == plan.len()
    {
        return left.reshaped(shape, policy);
    }
    if left
        .uniform_value()
        .is_some_and(|value| value.as_ref().is_some_and(Rational::is_zero))
        && operator == Op::Add
        && right.count(policy)? == plan.len()
    {
        return right.reshaped(shape, policy);
    }
    if let (Some(a), Some(b)) = (left.uniform_value(), right.uniform_value()) {
        let value = match (a, b) {
            (Some(a), Some(b)) => {
                Some(scalar(operator, a.clone(), b.clone(), policy, cooperation).await?)
            }
            _ => None,
        };
        return Ok(Data::uniform(shape, value));
    }
    let mut values = data::buffer(plan.len(), policy)?;
    for (_, a, b) in plan.iter() {
        values.push(match (left.at(a, policy)?, right.at(b, policy)?) {
            (Some(a), Some(b)) => Some(scalar(operator, a, b, policy, cooperation).await?),
            _ => None,
        });
        cooperation.completed(1).await?;
    }
    Data::explicit(shape, values, policy)
}
async fn matrix(
    left: Data,
    right: Data,
    shape: SharedShape,
    complete: bool,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Data> {
    let m = left.shape.first().copied().unwrap_or(1);
    let k = left.shape.get(1).copied().unwrap_or(1);
    let inner = right.shape.first().copied().unwrap_or(1);
    let n = right.shape.get(1).copied().unwrap_or(1);
    if k != inner || m.checked_mul(n) != Some(evaluation::element_count(&shape, policy)?) {
        return Err(policy.invalid("exact matrix geometry differs from original source"));
    }

    if complete
        && [left.uniform_value(), right.uniform_value()]
            .into_iter()
            .any(|value| value.is_some_and(|value| value.as_ref().is_some_and(Rational::is_zero)))
    {
        return Ok(Data::uniform(shape, Some(Rational::zero())));
    }
    if complete && (left.sparse().is_some() || right.sparse().is_some()) {
        return sparse::multiply(left, right, shape, m, k, n, policy, cooperation).await;
    }
    let mut values = data::buffer(evaluation::element_count(&shape, policy)?, policy)?;
    for column in 0..n {
        for row in 0..m {
            let mut sum = Some(Rational::zero());
            for axis in 0..k {
                sum = match (
                    sum,
                    left.at(row + axis * m, policy)?,
                    right.at(axis + column * k, policy)?,
                ) {
                    (Some(sum), Some(a), Some(b)) => Some(
                        sum.add(&a.mul(&b, policy, cooperation).await?, policy, cooperation)
                            .await?,
                    ),
                    _ => None,
                };
                cooperation.completed(1).await?;
            }
            values.push(sum);
        }
    }
    Data::explicit(shape, values, policy)
}
pub(super) async fn unary(
    input: Data,
    operator: Op,
    shape: SharedShape,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Data> {
    if operator == Op::UnaryPlus {
        return input.reshaped(shape, policy);
    }
    let count = input.count(policy)?;
    if operator == Op::UnaryMinus {
        if let Some(value) = input.uniform_value() {
            return Ok(Data::uniform(shape, value.as_ref().map(|value| -value)));
        }
    }
    let mut values = data::buffer(count, policy)?;
    let n = shape.first().copied().unwrap_or(1);
    let m = input.shape.first().copied().unwrap_or(1);
    for row in 0..count {
        let value = match operator {
            Op::UnaryMinus => input.at(row, policy)?.map(|value| -value),
            Op::Transpose | Op::ConjugateTranspose => input.at(row / n + (row % n) * m, policy)?,
            _ => return Err(policy.invalid("unadmitted exact unary operation")),
        };
        values.push(value);
        cooperation.completed(1).await?;
    }
    Data::explicit(shape, values, policy)
}
pub(super) async fn sum(
    input: Data,
    arguments: &[Value],
    shape: SharedShape,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Data> {
    let mut geometry = crate::builtins::math::reduction::sum::reduction_geometry(
        &input.shape,
        crate::call::FunctionArgumentView::from_slice(arguments),
        cooperation.runtime(),
    )
    .await?;
    if !crate::builtins::common::shape::array_shapes_equal(geometry.output_shape(), &shape) {
        return Err(policy.invalid("exact reduction differs from original source geometry"));
    }
    let count = evaluation::element_count(&shape, policy)?;
    if geometry.factor() == 0 {
        return Ok(Data::uniform(shape, Some(Rational::zero())));
    }
    if let Some(value) = input.uniform_value() {
        return Ok(Data::uniform(
            shape,
            match value {
                Some(value) => Some(
                    value
                        .mul(
                            &Rational::from_usize(geometry.factor()),
                            policy,
                            cooperation,
                        )
                        .await?,
                ),
                None => None,
            },
        ));
    }
    let mut values = data::buffer(count, policy)?;
    for _ in 0..count {
        values.push(Some(Rational::zero()));
        cooperation.completed(1).await?;
    }
    for row in 0..input.count(policy)? {
        let target = geometry
            .project(row)
            .ok_or_else(|| policy.invalid("exact reduction coordinate is unavailable"))?;
        values[target] = match (values[target].take(), input.at(row, policy)?) {
            (Some(sum), Some(value)) => Some(sum.add(&value, policy, cooperation).await?),
            _ => None,
        };
        cooperation.completed(1).await?;
    }
    Data::explicit(shape, values, policy)
}
