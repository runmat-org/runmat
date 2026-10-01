use super::*;

pub(crate) async fn icdf_probability_distribution(
    distribution: Value,
    p: Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Value> {
    check_icdf_cancellation(runtime)?;
    let fit = distribution_from_value(&distribution)?;
    let x = value_to_tensor("icdf", p).await?;
    let shape = x.shape.clone();
    let data = tensor::tensor_into_values_f64(x)
        .into_iter()
        .map(|value| icdf_scalar(&fit, value, || check_icdf_cancellation(runtime)))
        .collect::<BuiltinResult<Vec<_>>>()?;
    finish_for("icdf", shape, data)
}

fn check_icdf_cancellation(runtime: Option<&crate::context::RuntimeContext>) -> BuiltinResult<()> {
    if let Some(runtime) = runtime.filter(|runtime| runtime.is_cancelled()) {
        return Err(build_runtime_error("icdf: execution was cancelled")
            .with_identifier(format!("{}:Cancelled", runtime.error_namespace()))
            .with_builtin("icdf")
            .build());
    }
    Ok(())
}

/// Authenticate a fitted distribution before an `icdf` caller accesses a
/// resident probability input. Parsing remains owned by this domain.
pub(crate) fn validate_icdf_probability_distribution(value: &Value) -> BuiltinResult<()> {
    distribution_from_value(value).map(|_| ())
}
