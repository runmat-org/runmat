use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use runmat_value::{Tensor, Value};

use super::super::gpu_helpers;
use super::upload::upload_value_like_protected;
use crate::{build_runtime_error, BuiltinResult};

pub(crate) fn validate_real_unary_provider_output(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    output: GpuTensorHandle,
    builtin: &str,
) -> BuiltinResult<Value> {
    validate_real_unary_provider_output_protecting(provider, input, output, builtin, &[])
}

pub(crate) fn validate_real_unary_provider_output_protecting(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    output: GpuTensorHandle,
    builtin: &str,
    protected: &[GpuTensorHandle],
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
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
        },
    );
    let borrowed: Vec<_> = std::iter::once(input).chain(protected.iter()).collect();
    if !valid
        || borrowed
            .iter()
            .any(|handle| handle.shares_allocation_with(&output))
    {
        return Err(gpu_helpers::reject_gpu_output(
            output,
            build_runtime_error(format!(
                "{builtin}: provider returned malformed unary output"
            ))
            .with_builtin(builtin)
            .with_identifier(format!("RunMat:{builtin}:Internal"))
            .build(),
        ));
    }
    let output = gpu_helpers::propagate_owned_output_provenance(output, [input])?;
    gpu_helpers::resident_gpu_value(output)
}

pub(crate) async fn gather_compute_restore<F>(
    handle: GpuTensorHandle,
    builtin: &str,
    compute: F,
) -> BuiltinResult<Value>
where
    F: FnOnce(Tensor) -> BuiltinResult<Value>,
{
    gather_compute_restore_protecting(handle, builtin, &[], compute).await
}

pub(crate) async fn gather_compute_restore_protecting<F>(
    handle: GpuTensorHandle,
    builtin: &str,
    protected: &[GpuTensorHandle],
    compute: F,
) -> BuiltinResult<Value>
where
    F: FnOnce(Tensor) -> BuiltinResult<Value>,
{
    let provider = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
        build_runtime_error(format!("{builtin}: GPU input has no owning provider"))
            .with_builtin(builtin)
            .build()
    })?;
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    let output = compute(tensor)?;
    upload_value_like_protected(provider, output, builtin, &handle, protected)
}

pub(crate) async fn gather_value_compute_restore<F>(
    handle: GpuTensorHandle,
    builtin: &str,
    compute: F,
) -> BuiltinResult<Value>
where
    F: FnOnce(Value) -> BuiltinResult<Value>,
{
    gather_value_compute_restore_protecting(handle, builtin, &[], compute).await
}

pub(crate) async fn gather_value_compute_restore_protecting<F>(
    handle: GpuTensorHandle,
    builtin: &str,
    protected: &[GpuTensorHandle],
    compute: F,
) -> BuiltinResult<Value>
where
    F: FnOnce(Value) -> BuiltinResult<Value>,
{
    let provider = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
        build_runtime_error(format!("{builtin}: GPU input has no owning provider"))
            .with_builtin(builtin)
            .build()
    })?;
    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone())).await?;
    let output = compute(gathered)?;
    upload_value_like_protected(provider, output, builtin, &handle, protected)
}
