use runmat_builtins::TYPECAST_ERROR_INTERNAL;
use runmat_value::Value;

use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;

use super::{error, host, target::OutputTarget, BUILTIN_NAME};

pub(super) async fn execute(
    handle: runmat_accelerate_api::GpuTensorHandle,
    target: OutputTarget,
) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_storage(&handle)
        == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        || runmat_accelerate_api::handle_is_logical(&handle)
    {
        return Err(error::terminal_gpu(
            "complex and logical gpuArray inputs are not supported",
        ));
    }
    let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone()))
        .await
        .map_err(|cause| error::build(&TYPECAST_ERROR_INTERNAL, cause.message()))?;
    let result = host::reinterpret(gathered, target)?;
    gpu_helpers::restore_class_preserving_value(&handle, result, BUILTIN_NAME)
        .map_err(|cause| error::build(&TYPECAST_ERROR_INTERNAL, cause.message()))
}
