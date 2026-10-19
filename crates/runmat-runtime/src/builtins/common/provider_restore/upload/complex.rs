//! Complex restoration preserves native precision and protected input leases.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_value::ComplexTensor;

use crate::builtins::common::gpu_helpers;
use crate::{build_runtime_error, BuiltinResult};

pub(super) fn upload_complex_without_precision_override(
    provider: &dyn AccelProvider,
    tensor: &ComplexTensor,
    builtin: &str,
    protected: &[&GpuTensorHandle],
) -> BuiltinResult<GpuTensorHandle> {
    if tensor.integer_storage().is_some() {
        return Err(build_runtime_error(format!(
            "{builtin}: typed complex integer GPU buffers are not supported"
        ))
        .with_builtin(builtin)
        .build());
    }
    let handle = gpu_helpers::upload_complex_tensor_protecting(provider, tensor, protected)
        .map_err(|error| {
            build_runtime_error(format!(
                "{builtin}: failed to restore result to input provider: {error}"
            ))
            .with_builtin(builtin)
            .with_source(error)
            .build()
        })?;
    Ok(handle)
}
