use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_value::Value;

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin};
use crate::BuiltinResult;

use super::super::{errors, unary as host, BUILTIN_NAME};

pub(in super::super) async fn evaluate(input: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(&input)
        .ok_or_else(|| errors::internal("resident input has no exact owner"))?;
    let typed_fallback = runmat_accelerate_api::handle_integer_type(&input).is_some()
        || runmat_accelerate_api::handle_is_logical(&input);
    if !typed_fallback {
        match provider.unary_pow2(&input).await {
            Ok(mut output) if direct_output_matches(&output, &input, provider) => {
                gpu_helpers::propagate_output_provenance(&mut output, [&input])?;
                return gpu_helpers::resident_gpu_value(output);
            }
            Ok(output) => {
                return Err(gpu_helpers::reject_gpu_output(
                    output,
                    errors::internal("provider unary_pow2 returned malformed output"),
                ));
            }
            Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => {}
            Err(error) => {
                let mut failure = errors::internal(format!("provider unary_pow2 failed: {error}"));
                failure.source = Some(error.into_boxed_dyn_error());
                failure.gpu_gather_retry = crate::GpuGatherRetry::Never;
                return Err(failure);
            }
        }
    }
    fallback_and_restore(&input, provider).await
}

fn direct_output_matches(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    let contract = super::unary_contract(
        runmat_accelerate_api::handle_storage(input),
        runmat_accelerate_api::handle_precision(input),
    );
    gpu_helpers::unary_gpu_output_matches(output, input, provider, contract)
}

async fn fallback_and_restore(
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> BuiltinResult<Value> {
    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(input.clone()))
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let output = host::evaluate_host(gathered)?;
    let (mut restored, contract) = match &output {
        Value::Tensor(tensor) => (
            gpu_helpers::upload_tensor_protecting(provider, tensor, &[input])?,
            super::unary_contract(
                GpuTensorStorage::Real,
                super::precision_for(tensor.numeric_dtype()),
            ),
        ),
        Value::ComplexTensor(tensor) => (
            gpu_helpers::upload_complex_tensor_protecting(provider, tensor, &[input])?,
            super::unary_contract(
                GpuTensorStorage::ComplexInterleaved,
                super::precision_for(tensor.numeric_dtype()),
            ),
        ),
        _ => return Err(super::unsupported_output_kind()),
    };
    if !gpu_helpers::unary_gpu_output_matches(&restored, input, provider, contract) {
        return Err(gpu_helpers::reject_gpu_output(
            restored,
            errors::internal("provider upload returned malformed fallback output"),
        ));
    }
    gpu_helpers::propagate_output_provenance(&mut restored, [input])?;
    gpu_helpers::resident_gpu_value(restored)
}
