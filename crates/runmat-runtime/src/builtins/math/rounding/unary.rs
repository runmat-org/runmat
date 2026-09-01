use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_value::{Tensor, Value};

use crate::builtins::common::{gpu_helpers, tensor};
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
    let valid = gpu_helpers::unary_gpu_output_matches(
        &output,
        input,
        provider,
        gpu_helpers::UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision: runmat_accelerate_api::handle_precision(input),
            integer: None,
            logical: false,
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinct,
        },
    );
    if !valid {
        gpu_helpers::free_rejected_provider_output(&output, &[input], provider);
        return Err(error_with_detail(
            builtin,
            internal_error,
            "provider returned malformed unary output",
        ));
    }
    let mut output = output;
    runmat_accelerate_api::set_handle_provenance(
        &mut output,
        runmat_accelerate_api::handle_provenance(input)
            .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic),
    );
    Ok(gpu_helpers::resident_gpu_value(output))
}

pub(super) fn restore_tensor(
    provider: Option<&'static dyn AccelProvider>,
    tensor_value: Tensor,
    builtin: &str,
    internal_error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<Value> {
    let Some(provider) = provider else {
        return Ok(tensor::tensor_into_value(tensor_value));
    };
    let output = gpu_helpers::upload_tensor(provider, &tensor_value)
        .map_err(|error| error_with_detail(builtin, internal_error, error))?;
    Ok(gpu_helpers::resident_gpu_value(output))
}
