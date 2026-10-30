//! Resident/host scalar division routes and their post-effect metadata checks.

use super::{divide, DivisionContext};
use crate::builtins::common::gpu_helpers;
use crate::builtins::math::elementwise::binary_arithmetic::provider_support::{
    float_handle_supported, valid_real_binary_output,
};
use crate::BuiltinResult;
use runmat_accelerate_api::GpuTensorHandle;
use runmat_value::Value;

mod admission;
use admission::native_integer_division_supported;

pub(in super::super) async fn try_resident_numerator(
    context: DivisionContext,
    numerator: &GpuTensorHandle,
    denominator: &Value,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let provider = gpu_helpers::require_resident_compute_owner(numerator, || {
        context.provider_contract_error("resident numerator has no exact owning provider")
    })?;
    let Value::Num(scalar) = denominator else {
        return Ok(None);
    };
    if runmat_accelerate_api::handle_integer_type(numerator).is_some() {
        if !native_integer_division_supported(provider, context, numerator, false) {
            return Ok(None);
        }
        let Some(uploaded) = gpu_helpers::upload_exact_integer_scalar_like(
            provider, numerator, *scalar,
        )
        .map_err(|source| {
            let mut error = context.provider_contract_error(&source);
            error.source = Some(Box::new(source));
            error
        })?
        else {
            return Ok(None);
        };
        let uploaded_owner = uploaded;
        let uploaded = &uploaded_owner;
        let result = match divide(provider, numerator, uploaded).await {
            Ok(output) => Some(output),
            Err(error) if error.is_unsupported() => {
                return Err(context.provider_error_after_effect(error));
            }
            Err(error) => {
                return Err(context.provider_error_after_effect(error));
            }
        };
        let valid = result.as_ref().is_some_and(|output| {
            valid_real_binary_output(
                output,
                numerator,
                Some(uploaded),
                provider,
                numerator.shape(),
            )
        });
        if result.is_none() {
            return Ok(None);
        }
        if valid {
            return Ok(result);
        }
        return Err(gpu_helpers::reject_gpu_output(
            result.expect("validated present result"),
            context.provider_contract_error(
                "provider returned invalid element-wise division metadata",
            ),
        ));
    }
    if float_handle_supported(numerator) {
        let result = {
            let _metadata = gpu_helpers::ResidencyRestoreGuard::new(numerator);
            provider.scalar_div(numerator, *scalar)
        };
        match result {
            Ok(output) => {
                if valid_real_binary_output(&output, numerator, None, provider, numerator.shape()) {
                    return Ok(Some(output));
                }
                return Err(gpu_helpers::reject_gpu_output(
                    output,
                    context.provider_contract_error(
                        "provider returned invalid scalar division metadata",
                    ),
                ));
            }
            Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => {}
            Err(error) => return Err(context.provider_error(error)),
        }
    }
    Ok(None)
}

pub(in super::super) async fn try_resident_denominator(
    context: DivisionContext,
    numerator: &Value,
    denominator: &GpuTensorHandle,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let provider = gpu_helpers::require_resident_compute_owner(denominator, || {
        context.provider_contract_error("resident denominator has no exact owning provider")
    })?;
    let Value::Num(scalar) = numerator else {
        return Ok(None);
    };
    if runmat_accelerate_api::handle_integer_type(denominator).is_some() {
        if !native_integer_division_supported(provider, context, denominator, true) {
            return Ok(None);
        }
        let Some(uploaded) =
            gpu_helpers::upload_exact_integer_scalar_like(provider, denominator, *scalar).map_err(
                |source| {
                    let mut error = context.provider_contract_error(&source);
                    error.source = Some(Box::new(source));
                    error
                },
            )?
        else {
            return Ok(None);
        };
        let uploaded_owner = uploaded;
        let uploaded = &uploaded_owner;
        let result = match divide(provider, uploaded, denominator).await {
            Ok(output) => Some(output),
            Err(error) if error.is_unsupported() => {
                return Err(context.provider_error_after_effect(error));
            }
            Err(error) => {
                return Err(context.provider_error_after_effect(error));
            }
        };
        let valid = result.as_ref().is_some_and(|output| {
            valid_real_binary_output(
                output,
                denominator,
                Some(uploaded),
                provider,
                denominator.shape(),
            )
        });
        if result.is_none() {
            return Ok(None);
        }
        if valid {
            return Ok(result);
        }
        return Err(gpu_helpers::reject_gpu_output(
            result.expect("validated present result"),
            context.provider_contract_error(
                "provider returned invalid element-wise division metadata",
            ),
        ));
    }
    if float_handle_supported(denominator) {
        let result = {
            let _metadata = gpu_helpers::ResidencyRestoreGuard::new(denominator);
            provider.scalar_rdiv(denominator, *scalar)
        };
        match result {
            Ok(output) => {
                if valid_real_binary_output(
                    &output,
                    denominator,
                    None,
                    provider,
                    denominator.shape(),
                ) {
                    return Ok(Some(output));
                }
                return Err(gpu_helpers::reject_gpu_output(
                    output,
                    context.provider_contract_error(
                        "provider returned invalid scalar division metadata",
                    ),
                ));
            }
            Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => {}
            Err(error) => return Err(context.provider_error(error)),
        }
    }
    Ok(None)
}
