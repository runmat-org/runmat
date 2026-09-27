//! Restore host-computed results to the provider that owned a resident input.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_value::{IntegerStorage, Tensor, Value};

use super::{gpu_helpers, tensor};

#[cfg(test)]
mod tests;

/// The domain supplies every borrowed resident operand, including controls.
/// A rejected upload must never free an alias of any of those inputs.
pub(crate) fn restore_to_source_provider_protecting(
    value: Value,
    source: Option<&GpuTensorHandle>,
    protected: &[&GpuTensorHandle],
) -> Result<Value, String> {
    let Some(source) = source else {
        return Ok(value);
    };
    let provider = gpu_helpers::exact_provider_for_handle(source)
        .ok_or_else(|| "no acceleration provider is registered for resident output".to_string())?;
    let (tensor, logical) = value_as_tensor(value)?;
    let mut borrowed = protected.to_vec();
    borrowed.push(source);
    let mut handle = gpu_helpers::upload_tensor_protecting(provider, &tensor, &borrowed)?;
    if let Some(provenance) = runmat_accelerate_api::handle_provenance(source) {
        runmat_accelerate_api::set_handle_provenance(&mut handle, provenance);
    }
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
