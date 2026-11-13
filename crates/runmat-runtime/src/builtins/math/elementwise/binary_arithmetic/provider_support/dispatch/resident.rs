use runmat_accelerate_api::GpuTensorHandle;
use runmat_value::Value;

use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;

use super::super::{host_real_scalar, ArithmeticProviderOperation};
use super::common::{
    accept_output, float_handle_supported, provider_error, provider_error_after_effect,
};

#[path = "resident/mixed_add.rs"]
mod mixed_add;

#[cfg(all(test, feature = "wgpu", not(target_arch = "wasm32")))]
#[path = "resident/scalar_precision_tests.rs"]
mod scalar_precision_tests;

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
    let provider = gpu_helpers::require_resident_compute_owner(resident, || {
        super::common::provider_contract_error(
            operation,
            "resident input has no exact owning provider",
        )
    })?;
    // Original floating class precedes the legacy f64 scalar ABI. A mixed
    // Add decline goes directly to host execution, never a wrong-class scalar hook.
    if matches!(operation, ArithmeticProviderOperation::Add)
        && resident.physical().storage() == runmat_accelerate_api::GpuTensorStorage::Real
        && !resident.view().is_logical()
    {
        if let Some(control) = super::super::scalar::host_floating_scalar_literal(host) {
            use runmat_accelerate_api::{scalar_program::ScalarLiteral, NumericElementType};
            if matches!(
                (resident.physical().element_type(), control),
                (NumericElementType::F32, ScalarLiteral::F64Bits(_))
                    | (NumericElementType::F64, ScalarLiteral::F32Bits(_))
            ) {
                return mixed_add::try_add(provider, resident, host, control).await;
            }
        }
    }
    let Some(scalar) = host_real_scalar(host) else {
        return Ok(None);
    };
    if let Some(uploaded) = gpu_helpers::upload_exact_integer_scalar_like(
        provider, resident, scalar,
    )
    .map_err(|source| {
        crate::build_runtime_error(format!("{}: {source}", operation.name()))
            .with_builtin(operation.name())
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .with_source(source)
            .build()
    })? {
        let uploaded_owner = uploaded;
        let uploaded = &uploaded_owner;
        let result = if host_is_left {
            operation.apply(provider, uploaded, resident).await
        } else {
            operation.apply(provider, resident, uploaded).await
        };
        let accepted = match result {
            Ok(output) => accept_output(
                operation,
                output,
                resident,
                Some(uploaded),
                provider,
                resident.shape(),
            ),
            Err(error) => Err(provider_error_after_effect(operation, error)),
        };
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
        Ok(output) => accept_output(
            operation,
            output,
            resident,
            None,
            provider,
            resident.shape(),
        ),
        Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => Ok(None),
        Err(error) => Err(provider_error(operation, error)),
    }
}
