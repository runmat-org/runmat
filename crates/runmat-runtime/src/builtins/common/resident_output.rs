//! Restore host-computed results to the provider that owned a resident input.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_value::{IntegerStorage, Tensor, Value};

use super::gpu_helpers;
use crate::{build_runtime_error, BuiltinResult};

#[cfg(test)]
mod tests;

/// The domain supplies every borrowed resident operand, including controls.
/// A rejected upload must never free an alias of any of those inputs.
pub(crate) fn restore_to_source_provider_protecting(
    value: Value,
    source: Option<&GpuTensorHandle>,
    protected: &[&GpuTensorHandle],
) -> BuiltinResult<Value> {
    let Some(source) = source else {
        return Ok(value);
    };
    let provider = gpu_helpers::exact_provider_for_handle(source)
        .ok_or_else(|| "no acceleration provider is registered for resident output".to_string())?;
    let mut borrowed = protected.to_vec();
    borrowed.push(source);
    let logical = matches!(&value, Value::Bool(_) | Value::LogicalArray(_));
    let mut handle = match value {
        Value::Bool(value) => runmat_accelerate_api::upload_logical_protecting(
            provider,
            &runmat_accelerate_api::HostLogicalTensorView {
                data: &[u8::from(value)],
                shape: &[1, 1],
            },
            &borrowed,
        )
        .map_err(|source| {
            build_runtime_error(format!("cannot restore logical result: {source}"))
                .with_source(source)
                .build()
        })?,
        Value::LogicalArray(array) => runmat_accelerate_api::upload_logical_protecting(
            provider,
            &runmat_accelerate_api::HostLogicalTensorView {
                data: &array.data,
                shape: &array.shape,
            },
            &borrowed,
        )
        .map_err(|source| {
            build_runtime_error(format!("cannot restore logical result: {source}"))
                .with_source(source)
                .build()
        })?,
        value => {
            gpu_helpers::upload_tensor_protecting(provider, &value_as_tensor(value)?, &borrowed)?
        }
    };
    if let Some(provenance) = runmat_accelerate_api::handle_provenance(source) {
        handle = handle.with_provenance(provenance).map_err(|source| {
            build_runtime_error(format!("cannot restore resident intent: {source}"))
                .with_source(source)
                .build()
        })?;
    }
    if logical {
        gpu_helpers::logical_gpu_value(handle)
    } else {
        gpu_helpers::resident_gpu_value(handle)
    }
}

fn value_as_tensor(value: Value) -> Result<Tensor, String> {
    match value {
        Value::Int(value) => Tensor::new_integer(IntegerStorage::from_scalar(value), vec![1, 1]),
        Value::Num(value) => Tensor::new(vec![value], vec![1, 1]),
        Value::Tensor(tensor) => Ok(tensor),
        other => Err(format!("cannot restore resident result {other:?}")),
    }
}
