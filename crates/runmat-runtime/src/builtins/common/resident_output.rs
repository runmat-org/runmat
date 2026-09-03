//! Restore host-computed results to the provider that owned a resident input.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_value::{IntegerStorage, Tensor, Value};

use super::{gpu_helpers, tensor};

pub(crate) fn restore_to_source_provider(
    value: Value,
    source: Option<&GpuTensorHandle>,
) -> Result<Value, String> {
    let Some(source) = source else {
        return Ok(value);
    };
    let provider = runmat_accelerate_api::provider_for_handle(source)
        .or_else(runmat_accelerate_api::provider)
        .ok_or_else(|| "no acceleration provider is registered for resident output".to_string())?;
    let (tensor, logical) = value_as_tensor(value)?;
    let handle = gpu_helpers::upload_tensor(provider, &tensor)?;
    Ok(if logical {
        gpu_helpers::logical_gpu_value(handle)
    } else {
        gpu_helpers::resident_gpu_value(handle)
    })
}

fn value_as_tensor(value: Value) -> Result<(Tensor, bool), String> {
    match value {
        Value::Int(value) => Ok((
            Tensor::new_integer(IntegerStorage::from_scalar(value), vec![1, 1])?,
            false,
        )),
        Value::Num(value) => Ok((Tensor::new(vec![value], vec![1, 1])?, false)),
        Value::Tensor(tensor) => Ok((tensor, false)),
        Value::Bool(value) => Ok((
            Tensor::new(vec![f64::from(u8::from(value))], vec![1, 1])?,
            true,
        )),
        Value::LogicalArray(array) => Ok((tensor::logical_to_tensor(&array)?, true)),
        other => Err(format!("cannot restore resident result {other:?}")),
    }
}
