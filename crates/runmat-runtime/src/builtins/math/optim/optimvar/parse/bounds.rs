//! Bounds preserve original double buffers; uniform scalars stay constant size.
use super::*;
pub(in crate::builtins::math::optim::optimvar) async fn parse(
    value: &Value,
    shape: &[usize],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Bound> {
    let result = match value {
        Value::Num(value) => Bound::Uniform(*value),
        Value::Tensor(array) if array.as_f64_slice().is_some() => {
            if array.len() == 1 {
                Bound::Uniform(array.as_f64_slice().expect("double")[0])
            } else {
                // Bounds are elementwise values. A row/column orientation may
                // differ from a vector variable, but N-D extents must agree.
                let count = shape
                    .iter()
                    .try_fold(1usize, |size, &n| size.checked_mul(n))
                    .ok_or_else(|| invalid("bound shape overflow"))?;
                let vector = |shape: &[usize]| shape.len() == 2 && (shape[0] == 1 || shape[1] == 1);
                if array.len() != count
                    || (array.shape() != shape && !(vector(shape) && vector(array.shape())))
                {
                    return Err(invalid(
                        "bound arrays must match the optimization-variable dimensions",
                    ));
                }
                Bound::PerElement(
                    runmat_value::optimization_variable::OptimizationBoundArray::try_from_tensor(
                        array.clone(),
                    )
                    .map_err(|_| invalid("bounds must be full real double arrays"))?,
                )
            }
        }
        _ => return Err(invalid("bounds must be full real double arrays or scalars")),
    };
    match &result {
        Bound::Uniform(value) => {
            if value.is_nan() {
                return Err(invalid("bounds cannot contain NaN"));
            }
        }
        Bound::PerElement(values) => {
            for &value in values.as_f64_slice().expect("double buffer") {
                if value.is_nan() {
                    return Err(invalid("bounds cannot contain NaN"));
                }
                cooperation.completed(1).await?;
            }
        }
    }
    Ok(result)
}
