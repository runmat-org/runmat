use super::*;

pub(super) async fn corrcoef_host(
    args: CorrcoefArgs,
    requested_outputs: Option<usize>,
) -> BuiltinResult<Value> {
    let CorrcoefArgs {
        first,
        second,
        rows,
        alpha,
    } = args;
    let left = value_to_numeric_input_gather(first).await?;
    let right = match second {
        Some(value) => Some(value_to_numeric_input_gather(value).await?),
        None => None,
    };
    let evaluation = evaluate_inputs(left, right, rows)?;
    evaluation.outputs(requested_outputs, alpha)
}

async fn value_to_numeric_input_gather(value: Value) -> BuiltinResult<NumericInput> {
    let gathered = match value {
        Value::GpuTensor(handle) => {
            gpu_helpers::gather_value_async(&Value::GpuTensor(handle)).await?
        }
        other => other,
    };
    match gathered {
        Value::Complex(real, imag) => ComplexTensor::new(vec![(real, imag)], vec![1, 1])
            .map(NumericInput::Complex)
            .map_err(corrcoef_internal_error),
        Value::ComplexTensor(tensor) => Ok(NumericInput::Complex(tensor)),
        other => tensor::value_into_tensor_for("corrcoef", other)
            .map(NumericInput::Real)
            .map_err(corrcoef_internal_error),
    }
}
