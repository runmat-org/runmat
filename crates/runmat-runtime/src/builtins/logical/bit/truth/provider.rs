use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::LogicalBinaryOperator;
use runmat_value::{LogicalArray, Value};

use crate::builtins::common::gpu_helpers;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry};

mod errors;
mod output;

pub(super) fn select_binary_source(
    values: [&Value; 2],
    builtin: &str,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    gpu_helpers::select_resident_output_source(
        values.into_iter().filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        }),
        builtin,
    )
}

pub(super) fn select_unary_source(
    value: &Value,
    builtin: &str,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    gpu_helpers::select_resident_output_source(
        match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        },
        builtin,
    )
}

pub(super) fn binary_hook(
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
    builtin: &str,
    operation: LogicalBinaryOperator,
) -> BuiltinResult<Option<Value>> {
    if !floating_real(lhs) || !floating_real(rhs) || lhs.device_id != rhs.device_id {
        return Ok(None);
    }
    let owner = gpu_helpers::exact_provider_for_handle(lhs)
        .ok_or_else(|| errors::execution(builtin, "no provider owns the left operand"))?;
    let rhs_owner = gpu_helpers::exact_provider_for_handle(rhs)
        .ok_or_else(|| errors::execution(builtin, "no provider owns the right operand"))?;
    if !std::ptr::eq(owner, rhs_owner)
        || runmat_accelerate_api::handle_precision(lhs)
            != runmat_accelerate_api::handle_precision(rhs)
    {
        return Ok(None);
    }
    let result = match operation {
        LogicalBinaryOperator::And => owner.logical_and(lhs, rhs),
        LogicalBinaryOperator::Or => owner.logical_or(lhs, rhs),
        LogicalBinaryOperator::Xor => owner.logical_xor(lhs, rhs),
    };
    let mut output = match result {
        Ok(output) => output,
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => return Ok(None),
        Err(error) => {
            return Err(errors::execution(
                builtin,
                format!("provider execution failed: {error}"),
            ));
        }
    };
    if !output::valid_binary(&output, lhs, rhs, owner, builtin) {
        output::free_rejected(&output, &[lhs, rhs]);
        return Err(errors::payload(
            builtin,
            "provider returned an invalid logical output",
        ));
    }
    output::annotate(&mut output, [lhs, rhs]);
    Ok(Some(gpu_helpers::logical_gpu_value(output)))
}

pub(super) fn unary_hook(input: &GpuTensorHandle, builtin: &str) -> BuiltinResult<Option<Value>> {
    if !floating_real(input) {
        return Ok(None);
    }
    let owner = gpu_helpers::exact_provider_for_handle(input)
        .ok_or_else(|| errors::execution(builtin, "no provider owns the input operand"))?;
    let mut output = match owner.logical_not(input) {
        Ok(output) => output,
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => return Ok(None),
        Err(error) => {
            return Err(errors::execution(
                builtin,
                format!("provider execution failed: {error}"),
            ));
        }
    };
    if output.shape != input.shape || !output::valid(&output, input, owner) {
        output::free_rejected(&output, &[input]);
        return Err(errors::payload(
            builtin,
            "provider returned an invalid logical output",
        ));
    }
    output::annotate(&mut output, [input]);
    Ok(Some(gpu_helpers::logical_gpu_value(output)))
}

pub(super) fn restore_explicit(
    value: Value,
    source: Option<&GpuTensorHandle>,
    builtin: &str,
) -> BuiltinResult<Value> {
    let Some(source) = source.filter(|handle| runmat_accelerate_api::handle_is_explicit(handle))
    else {
        return Ok(value);
    };
    let value = match value {
        Value::Bool(bit) => Value::LogicalArray(
            LogicalArray::new(vec![u8::from(bit)], vec![1, 1]).map_err(|error| {
                build_runtime_error(format!("{builtin}: invalid scalar logical result: {error}"))
                    .with_builtin(builtin)
                    .build()
            })?,
        ),
        value => value,
    };
    let restored = gpu_helpers::restore_class_preserving_value(source, value, builtin)?;
    if !matches!(restored, Value::GpuTensor(_)) {
        return Err(build_runtime_error(format!(
            "{builtin}: provider cannot preserve explicit gpuArray output residency"
        ))
        .with_builtin(builtin)
        .with_identifier(format!("RunMat:{builtin}:GpuUploadFailed"))
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .build());
    }
    Ok(restored)
}

fn floating_real(handle: &GpuTensorHandle) -> bool {
    runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(handle).is_none()
}

#[cfg(test)]
mod tests;
