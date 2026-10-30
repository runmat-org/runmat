use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_value::Value;

use crate::builtins::common::{binary, gpu_helpers};
use crate::BuiltinResult;

use super::super::errors;

pub(in super::super) fn try_direct(
    significand: &GpuTensorHandle,
    exponent: &GpuTensorHandle,
) -> BuiltinResult<Option<Value>> {
    if !binary::matching_physical_inputs(significand, exponent)
        || runmat_accelerate_api::handle_integer_type(significand).is_some()
        || runmat_accelerate_api::handle_is_logical(significand)
    {
        return Ok(None);
    }
    let Some(provider) = shared_owner(significand, exponent) else {
        return Ok(None);
    };
    match provider.pow2_scale(significand, exponent) {
        Ok(output) => {
            let contract = gpu_helpers::BinaryGpuOutputContract {
                shape: significand.shape().to_vec(),
                storage: runmat_accelerate_api::handle_storage(significand),
                precision: runmat_accelerate_api::handle_precision(significand),
                integer: None,
                logical: false,
                alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
            };
            binary::validate_resident_output(provider, significand, exponent, output, &contract)
                .map(Some)
                .map_err(|source| {
                    let mut error = errors::internal(source.message());
                    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                    error.source = Some(Box::new(source));
                    error
                })
        }
        Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => Ok(None),
        Err(error) => Err(errors::internal(format!(
            "provider pow2_scale failed: {error}"
        ))),
    }
}

fn shared_owner(
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
) -> Option<&'static dyn AccelProvider> {
    let left_owner = gpu_helpers::exact_provider_for_handle(left)?;
    let right_owner = gpu_helpers::exact_provider_for_handle(right)?;
    (runmat_accelerate_api::same_provider_instance(left_owner, right_owner)
        && left.device_id() == right.device_id())
    .then_some(left_owner)
}
