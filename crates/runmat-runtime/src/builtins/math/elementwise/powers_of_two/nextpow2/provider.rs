use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, IntegerElementType, ProviderPrecision,
};
use runmat_value::Value;

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin};
use crate::BuiltinResult;

use super::{errors, operation, BUILTIN_NAME};

pub(super) async fn evaluate(input: GpuTensorHandle) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_storage(&input) == GpuTensorStorage::ComplexInterleaved {
        return Err(errors::invalid("complex gpuArray input is not supported"));
    }
    let provider = gpu_helpers::require_resident_compute_owner(&input, || {
        errors::internal("GPU input has no owning provider")
    })?;
    if output_representation(&input).integer.is_some()
        || runmat_accelerate_api::handle_is_logical(&input)
    {
        return evaluate_and_restore(&input, provider).await;
    }

    match provider.unary_nextpow2(&input).await {
        Ok(output) if output_matches(&output, &input, provider, output_representation(&input)) => {
            let output = gpu_helpers::propagate_owned_output_provenance(output, [&input])?;
            gpu_helpers::resident_gpu_value(output)
        }
        Ok(output) => {
            let mut error = errors::internal("provider unary_nextpow2 returned malformed output");
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            Err(gpu_helpers::reject_gpu_output(output, error))
        }
        Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => {
            evaluate_and_restore(&input, provider).await
        }
        Err(source) => {
            let mut error = errors::internal(format!("provider unary_nextpow2 failed: {source}"));
            error = crate::runtime_error::with_provider_execution_source(error, source.into());
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            Err(error)
        }
    }
}

#[derive(Clone, Copy)]
struct OutputRepresentation {
    precision: Option<ProviderPrecision>,
    integer: Option<IntegerElementType>,
}

fn output_representation(input: &GpuTensorHandle) -> OutputRepresentation {
    let integer = runmat_accelerate_api::handle_integer_type(input);
    let precision = if integer.is_some() {
        None
    } else if runmat_accelerate_api::handle_is_logical(input) {
        Some(ProviderPrecision::F64)
    } else {
        runmat_accelerate_api::handle_precision(input)
    };
    OutputRepresentation { precision, integer }
}

async fn evaluate_and_restore(
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
) -> BuiltinResult<Value> {
    let gathered = gpu_helpers::gather_tensor_async(input)
        .await
        .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
    let result = operation::transform(gathered)?;
    let output = gpu_helpers::upload_tensor_protecting(provider, &result, &[input])?;
    if !output_matches(&output, input, provider, output_representation(input)) {
        let mut error = errors::internal("provider upload returned malformed fallback output");
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        return Err(gpu_helpers::reject_gpu_output(output, error));
    }
    let output = gpu_helpers::propagate_owned_output_provenance(output, [input])?;
    gpu_helpers::resident_gpu_value(output)
}

fn output_matches(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
    representation: OutputRepresentation,
) -> bool {
    gpu_helpers::unary_gpu_output_matches(
        output,
        input,
        provider,
        gpu_helpers::UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision: representation.precision,
            integer: representation.integer,
            logical: false,
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
        },
    )
}
