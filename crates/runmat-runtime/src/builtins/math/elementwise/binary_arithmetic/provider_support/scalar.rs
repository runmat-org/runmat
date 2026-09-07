use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::BuiltinCatalogIdentity;
use runmat_value::Value;

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::BuiltinResult;

pub(in crate::builtins::math::elementwise::binary_arithmetic) fn host_real_scalar(
    value: &Value,
) -> Option<f64> {
    match value {
        Value::Num(value) => Some(*value),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::Tensor(value)
            if tensor::is_scalar_tensor(value) && value.integer_storage().is_none() =>
        {
            Some(tensor::tensor_value_f64(value, 0))
        }
        Value::LogicalArray(value) if value.data.len() == 1 => {
            Some(if value.data[0] != 0 { 1.0 } else { 0.0 })
        }
        Value::CharArray(value) if value.rows * value.cols == 1 => Some(
            value
                .data
                .first()
                .map(|&character| character as u32 as f64)
                .unwrap_or(0.0),
        ),
        _ => None,
    }
}

pub(in crate::builtins::math::elementwise::binary_arithmetic) async fn device_real_scalar(
    identity: BuiltinCatalogIdentity,
    handle: &GpuTensorHandle,
) -> BuiltinResult<Option<f64>> {
    if !is_scalar_shape(&handle.shape)
        || runmat_accelerate_api::handle_integer_type(handle).is_some()
    {
        return Ok(None);
    }
    let tensor = gpu_helpers::gather_tensor_async(handle)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, identity.name))?;
    Ok(tensor::tensor_values_f64(&tensor).first().copied())
}

pub(in crate::builtins::math::elementwise::binary_arithmetic) fn is_scalar_shape(
    shape: &[usize],
) -> bool {
    shape.iter().copied().product::<usize>() <= 1
}

#[cfg(test)]
mod tests {
    use super::{host_real_scalar, is_scalar_shape};
    use runmat_value::{NumericStorage, Tensor, Value};

    #[test]
    fn integer_storage_does_not_enter_the_floating_scalar_path() {
        let integer = Tensor::from_numeric_storage(NumericStorage::U64(vec![u64::MAX]), vec![1, 1])
            .expect("integer scalar");
        assert_eq!(host_real_scalar(&Value::Tensor(integer)), None);

        let floating = Tensor::new(vec![3.5], vec![1, 1]).expect("floating scalar");
        assert_eq!(host_real_scalar(&Value::Tensor(floating)), Some(3.5));
    }

    #[test]
    fn recognizes_provider_scalar_shapes() {
        assert!(is_scalar_shape(&[]));
        assert!(is_scalar_shape(&[1, 1, 1]));
        assert!(!is_scalar_shape(&[1, 2]));
    }
}
