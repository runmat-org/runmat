//! Compatibility at the real-input/possibly-complex result boundary.
use crate::builtins::common::{gpu_helpers, provider_restore};
use crate::{build_runtime_error, BuiltinResult};
use runmat_accelerate_api::{GpuHandleProvenance, GpuTensorHandle};
use runmat_builtins::BuiltinExtensionDescriptor;
use runmat_value::{Tensor, Value};

pub(in crate::builtins::math::trigonometry) async fn compute(
    handle: GpuTensorHandle,
    builtin: &str,
    extension: &'static BuiltinExtensionDescriptor,
    compute: impl FnOnce(Tensor) -> BuiltinResult<Value>,
) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
        build_runtime_error(format!("{builtin}: GPU input has no owning provider"))
            .with_builtin(builtin)
            .with_identifier(format!("RunMat:{builtin}:Internal"))
            .build()
    })?;
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    let output = compute(tensor)?;
    if matches!(output, Value::Complex(_, _) | Value::ComplexTensor(_)) {
        // Automatic residency cannot introduce an explicit-gpuArray extension
        // restriction. Legacy unannotated handles retain their resident policy.
        if runmat_accelerate_api::handle_provenance(&handle) == Some(GpuHandleProvenance::Automatic)
        {
            return Ok(output);
        }
        crate::compatibility::ensure_builtin_extension_enabled(extension, builtin)?;
    }
    provider_restore::upload_value_like(provider, output, builtin, &handle)
}
