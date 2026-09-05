use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::{HYPOT_ERROR_INTERNAL, HYPOT_ERROR_INVALID_INPUT};
use runmat_value::Value;

use crate::builtins::common::{binary as common_binary, gpu_helpers};
use crate::BuiltinResult;

use super::{errors, host, BUILTIN_NAME};

pub(super) async fn pair(left: GpuTensorHandle, right: GpuTensorHandle) -> BuiltinResult<Value> {
    let has_integer_input = runmat_accelerate_api::handle_integer_type(&left).is_some()
        || runmat_accelerate_api::handle_integer_type(&right).is_some();
    let provider = gpu_helpers::exact_provider_for_binary_inputs(&left, &right)
        .map_err(|error| errors::terminal(&HYPOT_ERROR_INVALID_INPUT, error))?;
    let real_floating = !has_integer_input
        && !runmat_accelerate_api::handle_is_logical(&left)
        && !runmat_accelerate_api::handle_is_logical(&right)
        && runmat_accelerate_api::handle_storage(&left) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_storage(&right) == GpuTensorStorage::Real;
    if real_floating && common_binary::matching_physical_inputs(&left, &right) {
        match provider.elem_hypot(&left, &right).await {
            Ok(handle) => {
                let contract = gpu_helpers::BinaryGpuOutputContract {
                    shape: left.shape.clone(),
                    storage: GpuTensorStorage::Real,
                    precision: runmat_accelerate_api::handle_precision(&left),
                    integer: None,
                    logical: false,
                    alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
                };
                return common_binary::validate_resident_output(
                    provider, &left, &right, handle, &contract,
                )
                .map_err(|error| errors::terminal(&HYPOT_ERROR_INTERNAL, error));
            }
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
            Err(error) => {
                return Err(errors::terminal(
                    &HYPOT_ERROR_INTERNAL,
                    format!("provider elem_hypot failed: {error}"),
                ));
            }
        }
    }
    let gathered_left = gather(&left).await?;
    let gathered_right = gather(&right).await?;
    super::admission::gathered_integer_boundary(&gathered_left)?;
    super::admission::gathered_integer_boundary(&gathered_right)?;
    let output = host::evaluate(gathered_left, gathered_right)?;
    let prototype = if runmat_accelerate_api::handle_is_explicit(&right)
        && !runmat_accelerate_api::handle_is_explicit(&left)
    {
        &right
    } else {
        &left
    };
    crate::builtins::common::provider_restore::upload_value_like_protected(
        provider,
        output,
        BUILTIN_NAME,
        prototype,
        &[left.clone(), right.clone()],
    )
}

pub(super) async fn mixed(
    handle: GpuTensorHandle,
    host: Value,
    gpu_is_left: bool,
) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
        errors::terminal(&HYPOT_ERROR_INTERNAL, "GPU provider unavailable for input")
    })?;
    let gathered = gather(&handle).await?;
    super::admission::gathered_integer_boundary(&gathered)?;
    let output = if gpu_is_left {
        host::evaluate(gathered, host)?
    } else {
        host::evaluate(host, gathered)?
    };
    crate::builtins::common::provider_restore::upload_value_like_protected(
        provider,
        output,
        BUILTIN_NAME,
        &handle,
        std::slice::from_ref(&handle),
    )
}

async fn gather(handle: &GpuTensorHandle) -> BuiltinResult<Value> {
    Ok(Value::Tensor(
        gpu_helpers::gather_tensor_async(handle).await?,
    ))
}
