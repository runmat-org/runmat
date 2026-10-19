use runmat_accelerate_api::{GpuTensorHandle, ProviderPrecision};
use runmat_value::Value;

use crate::builtins::common::{
    gpu_helpers, map_control_flow_with_builtin, provider_restore, tensor,
};
use crate::BuiltinResult;

use super::{errors, host, BUILTIN_NAME};

pub(super) async fn execute(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = runmat_accelerate_api::provider_for_handle(&handle)
        .ok_or_else(|| errors::internal("GPU input has no owning provider"))?;
    let integer = runmat_accelerate_api::handle_integer_type(&handle).is_some();
    let logical = runmat_accelerate_api::handle_is_logical(&handle);

    if !integer && !logical {
        match provider.unary_heaviside(&handle).await {
            Ok(output) => {
                return provider_restore::validate_real_unary_provider_output(
                    provider,
                    &handle,
                    output,
                    BUILTIN_NAME,
                )
            }
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
            Err(error) => return Err(errors::provider(error)),
        }
    }

    let downloaded = gpu_helpers::download_value_preserving_residency_async(provider, &handle)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let input =
        tensor::value_into_tensor_for(BUILTIN_NAME, downloaded).map_err(errors::invalid_input)?;
    let output = host::apply(input)?;

    if (integer || logical) && provider.precision() != ProviderPrecision::F64 {
        return Ok(tensor::tensor_into_value(output));
    }
    provider_restore::upload_value_like(provider, Value::Tensor(output), BUILTIN_NAME, &handle)
}
