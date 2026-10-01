use super::*;

pub(crate) async fn icdf_probability_distribution(
    distribution: Value,
    p: Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Value> {
    check_fitted_cancellation(runtime, FittedOperation::Inverse)?;
    let fit = distribution_from_value_for(FittedOperation::Inverse, &distribution)?;
    let x = value_to_tensor("icdf", p).await?;
    let shape = x.shape.clone();
    let data = tensor::tensor_into_values_f64(x)
        .into_iter()
        .map(|value| {
            icdf_scalar(&fit, value, || {
                check_fitted_cancellation(runtime, FittedOperation::Inverse)
            })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    finish_for("icdf", shape, data)
}

/// Authenticate a fitted distribution before an `icdf` caller accesses a
/// resident probability input. Parsing remains owned by this domain.
pub(crate) fn validate_icdf_probability_distribution(value: &Value) -> BuiltinResult<()> {
    distribution_from_value_for(FittedOperation::Inverse, value).map(|_| ())
}
