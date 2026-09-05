use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_value::{Tensor, Value};

use super::super::gpu_helpers;
use super::upload::upload_value_like;
use crate::{build_runtime_error, BuiltinResult};

pub(crate) fn validate_real_unary_provider_output(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    output: GpuTensorHandle,
    builtin: &str,
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
        return Err(build_runtime_error(format!(
            "{builtin}: provider returned malformed unary output"
        ))
        .with_builtin(builtin)
        .with_identifier(format!("RunMat:{builtin}:Internal"))
        .build());
    }
    let mut output = output;
    runmat_accelerate_api::set_handle_provenance(
        &mut output,
        runmat_accelerate_api::handle_provenance(input)
            .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic),
    );
    Ok(gpu_helpers::resident_gpu_value(output))
}

pub(crate) async fn gather_compute_restore<F>(
    handle: GpuTensorHandle,
    builtin: &str,
    compute: F,
) -> BuiltinResult<Value>
where
    F: FnOnce(Tensor) -> BuiltinResult<Value>,
{
    let provider = runmat_accelerate_api::provider_for_handle(&handle).ok_or_else(|| {
        build_runtime_error(format!("{builtin}: GPU input has no owning provider"))
            .with_builtin(builtin)
            .build()
    })?;
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    let output = compute(tensor)?;
    upload_value_like(provider, output, builtin, &handle)
}

pub(crate) async fn gather_value_compute_restore<F>(
    handle: GpuTensorHandle,
    builtin: &str,
    compute: F,
) -> BuiltinResult<Value>
where
    F: FnOnce(Value) -> BuiltinResult<Value>,
{
    let provider = runmat_accelerate_api::provider_for_handle(&handle).ok_or_else(|| {
        build_runtime_error(format!("{builtin}: GPU input has no owning provider"))
            .with_builtin(builtin)
            .build()
    })?;
    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone())).await?;
    let output = compute(gathered)?;
    upload_value_like(provider, output, builtin, &handle)
}
