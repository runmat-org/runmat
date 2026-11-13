//! Shared owning admission for nonconjugating and conjugating transpose.
use crate::builtins::common::gpu_helpers;
use crate::{BuiltinResult, RuntimeError};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, ProviderExecutionError};
use runmat_value::Value;

type ErrorMapper = fn(anyhow::Error) -> RuntimeError;

pub(in crate::builtins::math::linalg::ops) fn owner(
    input: &GpuTensorHandle,
    error: ErrorMapper,
) -> BuiltinResult<&'static dyn AccelProvider> {
    let provider = gpu_helpers::require_resident_compute_owner(input, || {
        error(anyhow::anyhow!(
            "resident input has no exact owning provider"
        ))
    })?;
    input
        .allocation_lease()
        .validate_live()
        .map_err(|cause| error(cause.into()))?;
    Ok(provider)
}

pub(in crate::builtins::math::linalg::ops) fn shape(input: &GpuTensorHandle) -> Vec<usize> {
    match input.shape() {
        [] => vec![],
        [rows] => vec![1, *rows],
        shape => {
            let mut shape = shape.to_vec();
            shape.swap(0, 1);
            shape
        }
    }
}

pub(in crate::builtins::math::linalg::ops) fn output_matches(
    input: &GpuTensorHandle,
    output: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
    shape: &[usize],
) -> bool {
    gpu_helpers::binary_gpu_output_matches(
        output,
        input,
        input,
        provider,
        &gpu_helpers::BinaryGpuOutputContract {
            shape: shape.to_vec(),
            storage: runmat_accelerate_api::handle_storage(input),
            precision: runmat_accelerate_api::handle_precision(input),
            integer: runmat_accelerate_api::handle_integer_type(input),
            logical: runmat_accelerate_api::handle_is_logical(input),
            // A validated immutable lazy transpose can own the same physical data.
            alias: gpu_helpers::GpuOutputAliasPolicy::AllowInput,
        },
    )
}

pub(in crate::builtins::math::linalg::ops) fn live(
    output: GpuTensorHandle,
    error: ErrorMapper,
) -> BuiltinResult<GpuTensorHandle> {
    if let Err(cause) = output.allocation_lease().validate_live() {
        return Err(gpu_helpers::reject_gpu_output(output, error(cause.into())));
    }
    Ok(output)
}

pub(in crate::builtins::math::linalg::ops) fn adopt(
    input: &GpuTensorHandle,
    output: GpuTensorHandle,
    error: ErrorMapper,
) -> BuiltinResult<Value> {
    let output = live(output, error)?;
    let output = gpu_helpers::propagate_owned_output_provenance(output, [input])
        .map_err(|cause| source_error(error, cause))?;
    gpu_helpers::resident_gpu_value(output).map_err(|cause| source_error(error, cause))
}

fn source_error(error: ErrorMapper, cause: RuntimeError) -> RuntimeError {
    // RuntimeError can retain platform-local ownership on WASM. Preserve its
    // source through the platform-aware carrier, never erase it into anyhow.
    crate::runtime_error::with_provider_execution_source(
        error(anyhow::anyhow!(cause.to_string())),
        ProviderExecutionError::from_source(cause),
    )
}
