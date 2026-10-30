use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_value::Value;

use crate::builtins::common::gpu_helpers;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(super) fn error_with_detail(
    builtin: &str,
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {detail}", error.message)).with_builtin(builtin);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(super) fn reject_excess_outputs(
    builtin: &str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<()> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
        return Err(error_with_detail(
            builtin,
            error,
            "only one output is supported",
        ));
    }
    Ok(())
}

pub(super) fn validate_provider_output(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    output: GpuTensorHandle,
    builtin: &str,
    internal_error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<Value> {
    validate_provider_output_with_storage(
        provider,
        input,
        output,
        GpuTensorStorage::Real,
        builtin,
        internal_error,
    )
}

pub(super) fn validate_provider_output_with_storage(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    output: GpuTensorHandle,
    storage: GpuTensorStorage,
    builtin: &str,
    internal_error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<Value> {
    let valid = gpu_helpers::numeric_descriptor_matches_source(input, &output)
        && gpu_helpers::unary_gpu_output_matches(
            &output,
            input,
            provider,
            gpu_helpers::UnaryGpuOutputContract {
                storage,
                precision: runmat_accelerate_api::handle_precision(input),
                integer: None,
                logical: false,
                alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
            },
        );
    if !valid {
        let mut error = error_with_detail(
            builtin,
            internal_error,
            "provider returned malformed unary output",
        );
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        return Err(gpu_helpers::reject_gpu_output(output, error));
    }
    let output = gpu_helpers::propagate_owned_output_provenance(output, [input])?;
    gpu_helpers::resident_gpu_value(output)
}

pub(super) fn input_provider(
    input: &GpuTensorHandle,
    builtin: &str,
    internal_error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<&'static dyn AccelProvider> {
    let provider = gpu_helpers::require_resident_compute_owner(input, || {
        error_with_detail(
            builtin,
            internal_error,
            "no acceleration provider owns the input handle",
        )
    })?;
    let element = gpu_helpers::expected_handle_numeric_element_type(input)
        .map_err(|error| error_with_detail(builtin, internal_error, error))?;
    if !gpu_helpers::numeric_descriptor_matches(
        input,
        element,
        runmat_accelerate_api::handle_storage(input),
    ) || (!input.shape().contains(&0)
        && input
            .shape()
            .iter()
            .try_fold(1usize, |count, extent| count.checked_mul(*extent))
            .is_none())
    {
        return Err(error_with_detail(
            builtin,
            internal_error,
            "malformed resident input descriptor",
        ));
    }
    Ok(provider)
}

/// Root a rounding result at its numerical or synchronous object publication.
pub(super) fn root_output(
    value: runmat_value::Value,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    crate::execution::RootedValue::new(value).map_err(|error| error.into_runtime_error())
}
