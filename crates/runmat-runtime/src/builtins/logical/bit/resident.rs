use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_value::{LogicalArray, Value};

use crate::builtins::common::broadcast::broadcast_shapes;
use crate::builtins::common::gpu_helpers;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry};

#[derive(Clone, Copy)]
pub(super) enum LogicalBinaryOp {
    Or,
    Xor,
}

pub(super) fn select_output_source(
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

pub(super) fn select_unary_output_source(
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

pub(super) fn try_binary_hook(
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
    builtin: &str,
    operation: LogicalBinaryOp,
) -> Option<Value> {
    if !floating_real_handle(lhs) || !floating_real_handle(rhs) || lhs.device_id != rhs.device_id {
        return None;
    }
    let owner = gpu_helpers::exact_provider_for_handle(lhs)?;
    let rhs_owner = gpu_helpers::exact_provider_for_handle(rhs)?;
    if !std::ptr::eq(owner, rhs_owner)
        || runmat_accelerate_api::handle_precision(lhs)
            != runmat_accelerate_api::handle_precision(rhs)
    {
        return None;
    }
    let mut output = match operation {
        LogicalBinaryOp::Or => owner.logical_or(lhs, rhs),
        LogicalBinaryOp::Xor => owner.logical_xor(lhs, rhs),
    }
    .ok()?;
    if !valid_binary_output(&output, lhs, rhs, owner, builtin) {
        free_rejected_output(&output, &[lhs, rhs]);
        return None;
    }
    annotate_logical_output(&mut output, [lhs, rhs]);
    Some(gpu_helpers::logical_gpu_value(output))
}

pub(super) fn try_unary_hook(input: &GpuTensorHandle) -> Option<Value> {
    if !floating_real_handle(input) {
        return None;
    }
    let owner = gpu_helpers::exact_provider_for_handle(input)?;
    let mut output = owner.logical_not(input).ok()?;
    if !valid_unary_output(&output, input, owner) {
        free_rejected_output(&output, &[input]);
        return None;
    }
    annotate_logical_output(&mut output, [input]);
    Some(gpu_helpers::logical_gpu_value(output))
}

pub(super) fn restore_explicit_logical_result(
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

fn floating_real_handle(handle: &GpuTensorHandle) -> bool {
    runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(handle).is_none()
}

fn valid_binary_output(
    output: &GpuTensorHandle,
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
    owner: &'static dyn AccelProvider,
    builtin: &str,
) -> bool {
    let expected_shape = broadcast_shapes(builtin, &lhs.shape, &rhs.shape).ok();
    expected_shape.as_deref() == Some(output.shape.as_slice())
        && valid_output(output, lhs, owner)
        && !same_handle(output, rhs)
}

fn valid_unary_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    owner: &'static dyn AccelProvider,
) -> bool {
    output.shape == input.shape && valid_output(output, input, owner)
}

fn valid_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    owner: &'static dyn AccelProvider,
) -> bool {
    output.device_id == input.device_id
        && !same_handle(output, input)
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(output)
            == runmat_accelerate_api::handle_precision(input)
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && gpu_helpers::exact_provider_for_handle(output)
            .is_some_and(|output_owner| std::ptr::eq(output_owner, owner))
}

fn annotate_logical_output<'a>(
    output: &mut GpuTensorHandle,
    inputs: impl IntoIterator<Item = &'a GpuTensorHandle>,
) {
    runmat_accelerate_api::set_handle_logical(output, true);
    let provenance = inputs
        .into_iter()
        .filter_map(runmat_accelerate_api::handle_provenance)
        .find(|provenance| *provenance == runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
    runmat_accelerate_api::set_handle_provenance(output, provenance);
    runmat_accelerate_api::mark_residency(output);
}

fn same_handle(lhs: &GpuTensorHandle, rhs: &GpuTensorHandle) -> bool {
    lhs.device_id == rhs.device_id && lhs.buffer_id == rhs.buffer_id
}

fn free_rejected_output(output: &GpuTensorHandle, protected: &[&GpuTensorHandle]) {
    if protected.iter().any(|handle| same_handle(output, handle)) {
        return;
    }
    if let Some(owner) = gpu_helpers::exact_provider_for_handle(output) {
        let _ = owner.free(output);
    }
}
