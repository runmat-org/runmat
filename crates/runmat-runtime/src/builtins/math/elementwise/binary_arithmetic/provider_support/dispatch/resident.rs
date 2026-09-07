use runmat_accelerate_api::GpuTensorHandle;
use runmat_value::Value;

use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;

use super::super::{host_real_scalar, ArithmeticProviderOperation};
use super::common::{accept_output, accepted_output, float_handle_supported, provider_error};

pub(in crate::builtins::math::elementwise::binary_arithmetic) async fn try_host_right(
    operation: ArithmeticProviderOperation,
    resident: &GpuTensorHandle,
    host: &Value,
) -> BuiltinResult<Option<Value>> {
    try_host_scalar(operation, resident, host, false).await
}

pub(in crate::builtins::math::elementwise::binary_arithmetic) async fn try_host_left(
    operation: ArithmeticProviderOperation,
    host: &Value,
    resident: &GpuTensorHandle,
) -> BuiltinResult<Option<Value>> {
    try_host_scalar(operation, resident, host, true).await
}

async fn try_host_scalar(
    operation: ArithmeticProviderOperation,
    resident: &GpuTensorHandle,
    host: &Value,
    host_is_left: bool,
) -> BuiltinResult<Option<Value>> {
    let Some(provider) = gpu_helpers::exact_provider_for_handle(resident) else {
        return Ok(None);
    };
    let Some(scalar) = host_real_scalar(host) else {
        return Ok(None);
    };
    if let Some(uploaded) =
        gpu_helpers::upload_exact_integer_scalar_like(provider, resident, scalar)
            .map_err(|error| provider_error(operation, error))?
    {
        let result = if host_is_left {
            operation.apply(provider, &uploaded, resident).await
        } else {
            operation.apply(provider, resident, &uploaded).await
        };
        let accepted = match result {
            Ok(output) => accept_output(
                operation,
                output,
                resident,
                Some(&uploaded),
                provider,
                &resident.shape,
            ),
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => Ok(None),
            Err(error) => Err(provider_error(operation, error)),
        };
        let protected = accepted_output(&accepted);
        let protected = protected
            .into_iter()
            .chain(std::iter::once(resident))
            .collect::<Vec<_>>();
        gpu_helpers::free_rejected_provider_output(&uploaded, &protected, provider);
        if accepted.as_ref().is_ok_and(Option::is_some) {
            return accepted;
        }
        accepted?;
    }
    if !float_handle_supported(resident) {
        return Ok(None);
    }
    let result = if host_is_left {
        operation.scalar_left(provider, resident, scalar)
    } else {
        operation.scalar_right(provider, resident, scalar)
    };
    match result {
        Ok(output) => accept_output(operation, output, resident, None, provider, &resident.shape),
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => Ok(None),
        Err(error) => Err(provider_error(operation, error)),
    }
}
