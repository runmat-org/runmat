use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_value::Value;

use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::common::gpu_helpers;
use crate::builtins::math::elementwise::binary_arithmetic::provider_support::valid_real_binary_output;
use crate::BuiltinResult;

use super::DivisionContext;

pub(super) async fn try_pair(
    context: DivisionContext,
    numerator: &GpuTensorHandle,
    denominator: &GpuTensorHandle,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let Some(provider) = common_owner(numerator, denominator) else {
        return Ok(None);
    };
    if !pair_metadata_supported(numerator, denominator) {
        return Ok(None);
    }
    let expected_shape = BroadcastPlan::new(&numerator.shape, &denominator.shape)
        .map_err(|detail| context.error_with_detail(context.size_mismatch, detail))?
        .output_shape()
        .to_vec();
    let output = match provider.elem_div(numerator, denominator).await {
        Ok(output) => output,
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => return Ok(None),
        Err(error) => return Err(context.internal_error(error)),
    };
    if valid_real_binary_output(
        &output,
        numerator,
        Some(denominator),
        provider,
        &expected_shape,
    ) {
        Ok(Some(output))
    } else {
        gpu_helpers::free_rejected_provider_output(&output, &[numerator, denominator], provider);
        Err(context.internal_error("provider returned invalid element-wise division metadata"))
    }
}

pub(super) async fn try_resident_numerator(
    context: DivisionContext,
    numerator: &GpuTensorHandle,
    denominator: &Value,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let Some(provider) = gpu_helpers::exact_provider_for_handle(numerator) else {
        return Ok(None);
    };
    let Value::Num(scalar) = denominator else {
        return Ok(None);
    };
    if runmat_accelerate_api::handle_integer_type(numerator).is_some() {
        let Some(uploaded) =
            gpu_helpers::upload_exact_integer_scalar_like(provider, numerator, *scalar)
                .map_err(|error| context.internal_error(error))?
        else {
            return Ok(None);
        };
        let result = match provider.elem_div(numerator, &uploaded).await {
            Ok(output) => Some(output),
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => None,
            Err(error) => {
                gpu_helpers::free_unprotected_exact_owner(&uploaded, &[numerator]);
                return Err(context.internal_error(error));
            }
        };
        let valid = result.as_ref().is_some_and(|output| {
            valid_real_binary_output(
                output,
                numerator,
                Some(&uploaded),
                provider,
                &numerator.shape,
            )
        });
        if let Some(output) = result.as_ref().filter(|_| !valid) {
            gpu_helpers::free_rejected_provider_output(output, &[numerator, &uploaded], provider);
        }
        gpu_helpers::free_unprotected_exact_owner(&uploaded, &[numerator]);
        if result.is_none() {
            return Ok(None);
        }
        if valid {
            return Ok(result);
        }
        return Err(
            context.internal_error("provider returned invalid element-wise division metadata")
        );
    }
    if float_handle_supported(numerator) {
        match provider.scalar_div(numerator, *scalar) {
            Ok(output) => {
                if valid_real_binary_output(&output, numerator, None, provider, &numerator.shape) {
                    return Ok(Some(output));
                }
                gpu_helpers::free_rejected_provider_output(&output, &[numerator], provider);
                return Err(
                    context.internal_error("provider returned invalid scalar division metadata")
                );
            }
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
            Err(error) => return Err(context.internal_error(error)),
        }
    }
    Ok(None)
}

pub(super) async fn try_resident_denominator(
    context: DivisionContext,
    numerator: &Value,
    denominator: &GpuTensorHandle,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let Some(provider) = gpu_helpers::exact_provider_for_handle(denominator) else {
        return Ok(None);
    };
    let Value::Num(scalar) = numerator else {
        return Ok(None);
    };
    if runmat_accelerate_api::handle_integer_type(denominator).is_some() {
        let Some(uploaded) =
            gpu_helpers::upload_exact_integer_scalar_like(provider, denominator, *scalar)
                .map_err(|error| context.internal_error(error))?
        else {
            return Ok(None);
        };
        let result = match provider.elem_div(&uploaded, denominator).await {
            Ok(output) => Some(output),
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => None,
            Err(error) => {
                gpu_helpers::free_unprotected_exact_owner(&uploaded, &[denominator]);
                return Err(context.internal_error(error));
            }
        };
        let valid = result.as_ref().is_some_and(|output| {
            valid_real_binary_output(
                output,
                denominator,
                Some(&uploaded),
                provider,
                &denominator.shape,
            )
        });
        if let Some(output) = result.as_ref().filter(|_| !valid) {
            gpu_helpers::free_rejected_provider_output(output, &[denominator, &uploaded], provider);
        }
        gpu_helpers::free_unprotected_exact_owner(&uploaded, &[denominator]);
        if result.is_none() {
            return Ok(None);
        }
        if valid {
            return Ok(result);
        }
        return Err(
            context.internal_error("provider returned invalid element-wise division metadata")
        );
    }
    if float_handle_supported(denominator) {
        match provider.scalar_rdiv(denominator, *scalar) {
            Ok(output) => {
                if valid_real_binary_output(
                    &output,
                    denominator,
                    None,
                    provider,
                    &denominator.shape,
                ) {
                    return Ok(Some(output));
                }
                gpu_helpers::free_rejected_provider_output(&output, &[denominator], provider);
                return Err(
                    context.internal_error("provider returned invalid scalar division metadata")
                );
            }
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
            Err(error) => return Err(context.internal_error(error)),
        }
    }
    Ok(None)
}

fn common_owner(
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
) -> Option<&'static dyn AccelProvider> {
    let left_owner = gpu_helpers::exact_provider_for_handle(left)?;
    let right_owner = gpu_helpers::exact_provider_for_handle(right)?;
    (std::ptr::eq(left_owner, right_owner) && left.device_id == right.device_id)
        .then_some(left_owner)
}

fn float_handle_supported(handle: &GpuTensorHandle) -> bool {
    runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(handle).is_none()
        && !runmat_accelerate_api::handle_is_logical(handle)
        && runmat_accelerate_api::handle_precision(handle).is_some()
}

fn pair_metadata_supported(left: &GpuTensorHandle, right: &GpuTensorHandle) -> bool {
    if runmat_accelerate_api::handle_storage(left) != GpuTensorStorage::Real
        || runmat_accelerate_api::handle_storage(right) != GpuTensorStorage::Real
        || runmat_accelerate_api::handle_is_logical(left)
        || runmat_accelerate_api::handle_is_logical(right)
    {
        return false;
    }
    match (
        runmat_accelerate_api::handle_integer_type(left),
        runmat_accelerate_api::handle_integer_type(right),
    ) {
        (Some(left), Some(right)) => left == right,
        (None, None) => {
            runmat_accelerate_api::handle_precision(left)
                == runmat_accelerate_api::handle_precision(right)
                && runmat_accelerate_api::handle_precision(left).is_some()
        }
        _ => false,
    }
}
