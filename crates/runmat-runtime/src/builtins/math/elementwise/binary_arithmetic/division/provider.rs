use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_value::Value;

use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::common::gpu_helpers;
use crate::builtins::math::elementwise::binary_arithmetic::provider_support::{
    common_owner, compatible_inputs, float_handle_supported, valid_real_binary_output,
};
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
    if !compatible_inputs(numerator, denominator) {
        return Ok(None);
    }
    let expected_shape = BroadcastPlan::new(numerator.shape(), denominator.shape())
        .map_err(|detail| context.error_with_detail(context.size_mismatch, detail))?
        .output_shape()
        .to_vec();
    let output = match divide(provider, numerator, denominator).await {
        Ok(output) => output,
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => return Ok(None),
        Err(error) => return Err(context.provider_error(error)),
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
        Err(gpu_helpers::reject_gpu_output(
            output,
            context.provider_contract_error(
                "provider returned invalid element-wise division metadata",
            ),
        ))
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
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => None,
            Err(error) => {
                return Err(context.provider_error(error));
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
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
            Err(error) => return Err(context.provider_error(error)),
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
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => None,
            Err(error) => {
                return Err(context.provider_error(error));
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
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
            Err(error) => return Err(context.provider_error(error)),
        }
    }
    Ok(None)
}

async fn divide(
    provider: &dyn AccelProvider,
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
) -> anyhow::Result<GpuTensorHandle> {
    let _left = gpu_helpers::ResidencyRestoreGuard::new(left);
    let _right = gpu_helpers::ResidencyRestoreGuard::new(right);
    provider.elem_div(left, right).await
}
