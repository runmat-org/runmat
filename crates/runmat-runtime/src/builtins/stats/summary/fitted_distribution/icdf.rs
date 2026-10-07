use super::*;

/// Validated fitted-object state carried across the caller's probability
/// admission and provider gather without reparsing serialized properties.
pub(crate) struct FittedDistributionAdmission(FittedDistribution);

pub(crate) fn admit_icdf_probability_distribution(
    value: &Value,
) -> BuiltinResult<FittedDistributionAdmission> {
    distribution_from_value_for(FittedOperation::Inverse, value).map(FittedDistributionAdmission)
}

pub(crate) async fn icdf_probability_distribution(
    admitted: FittedDistributionAdmission,
    p: Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Value> {
    check_fitted_cancellation(runtime, FittedOperation::Inverse)?;
    let fit = admitted.0;
    let x = value_to_tensor("icdf", p).await?;
    let shape = x.shape().to_vec();
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
