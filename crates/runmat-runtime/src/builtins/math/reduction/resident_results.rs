//! Transactional restoration for reduction operations with two host results.
use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_value::{IntegerStorage, Tensor, Value};

pub(super) fn restore_pair(
    source: &GpuTensorHandle,
    first: Value,
    second: Value,
    retained: usize,
    builtin: &str,
    protected: &[&GpuTensorHandle],
) -> BuiltinResult<(Value, Value)> {
    if retained == 0 {
        return Ok((first, second));
    }
    let first = restore(source, first, builtin, protected)?;
    if retained == 1 {
        return Ok((first, second));
    }
    let mut second_protected = protected.to_vec();
    second_protected.push(source);
    if let Value::GpuTensor(handle) = &first {
        second_protected.push(handle);
    }
    match restore(source, second, builtin, &second_protected) {
        Ok(second) => Ok((first, second)),
        Err(error) => {
            if let Value::GpuTensor(handle) = first {
                return Err(gpu_helpers::reject_gpu_output(handle, error));
            }
            Err(error)
        }
    }
}

fn restore(
    source: &GpuTensorHandle,
    value: Value,
    builtin: &str,
    protected: &[&GpuTensorHandle],
) -> BuiltinResult<Value> {
    // The shared class-preserving restorer handles numeric tensors. Preserve
    // exact scalar integers at this adapter boundary instead of promoting them.
    let (value, scalar) = match value {
        Value::Int(integer) => {
            let tensor =
                Tensor::new_integer(IntegerStorage::from_scalar(integer.clone()), vec![1, 1])
                    .map_err(|error| {
                        crate::build_runtime_error(format!("{builtin}: {error}")).build()
                    })?;
            (Value::Tensor(tensor), Some(integer))
        }
        other => (other, None),
    };
    let value =
        gpu_helpers::restore_class_preserving_value_protecting(source, value, builtin, protected)?;
    match (scalar, value) {
        (Some(integer), Value::Tensor(_)) => Ok(Value::Int(integer)),
        (_, value) => Ok(value),
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "resident_results/owning_tests.rs"]
mod owning_tests;
