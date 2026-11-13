//! Host numeric and structural value admission for empirical statistics.
use super::*;
use runmat_value::IntValue;

pub(super) fn numeric_vector(value: &Value, builtin: EmpiricalBuiltin) -> BuiltinResult<Vec<f64>> {
    let tensor = tensor_from_value(value.clone(), builtin)?;
    if tensor.rows() > 1 && tensor.cols() > 1 {
        return Err(invalid_argument(
            builtin,
            format!(
                "{builtin}: expected a vector, got {}x{} matrix",
                tensor.rows(),
                tensor.cols()
            ),
        ));
    }
    Ok(tensor_values_f64(&tensor))
}

pub(super) fn tensor_from_value(value: Value, builtin: EmpiricalBuiltin) -> BuiltinResult<Tensor> {
    match value {
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            Tensor::new(tensor_values_f64(&tensor), tensor.shape().to_vec())
                .map_err(|err| internal_error(builtin, format!("{builtin}: {err}")))
        }
        Value::Tensor(tensor) => Ok(tensor),
        Value::Num(value) => Tensor::new(vec![value], vec![1, 1])
            .map_err(|err| internal_error(builtin, format!("{builtin}: {err}"))),
        Value::Int(value) => Tensor::new(vec![value.to_f64()], vec![1, 1])
            .map_err(|err| internal_error(builtin, format!("{builtin}: {err}"))),
        Value::Bool(value) => Tensor::new(vec![if value { 1.0 } else { 0.0 }], vec![1, 1])
            .map_err(|err| internal_error(builtin, format!("{builtin}: {err}"))),
        Value::LogicalArray(array) => {
            let mut data = Vec::new();
            data.try_reserve_exact(array.data.len()).map_err(|cause| {
                let mut error = internal_error(builtin, format!("{builtin}: {cause}"));
                error.source = Some(Box::new(cause));
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                error
            })?;
            for &bit in &array.data {
                data.push(if bit == 0 { 0.0 } else { 1.0 });
            }
            Tensor::new(data, array.shape)
                .map_err(|err| internal_error(builtin, format!("{builtin}: {err}")))
        }
        other => Tensor::try_from(&other)
            .map_err(|err| invalid_argument(builtin, format!("{builtin}: {err}"))),
    }
}

pub(super) fn scalar_number(value: &Value) -> Option<f64> {
    if let Some(value) = integer_scalar(value) {
        return Some(value.to_f64());
    }
    match value {
        Value::Num(value) => Some(*value),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Some(tensor::tensor_value_f64(tensor, 0))
        }
        _ => None,
    }
}

pub(super) fn integer_scalar(value: &Value) -> Option<IntValue> {
    match value {
        Value::Int(value) => Some(value.clone()),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0)),
        _ => None,
    }
}

pub(super) fn tensor_values_f64(tensor: &Tensor) -> Vec<f64> {
    tensor::tensor_values_f64(tensor)
}

pub(super) fn column_tensor(data: Vec<f64>) -> BuiltinResult<Tensor> {
    let len = data.len();
    Tensor::new(data, vec![len, 1])
        .map_err(|err| internal_error(ECDF_OWNER, format!("ecdf: {err}")))
}
