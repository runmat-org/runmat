use runmat_builtins::{PERMS_ERROR_INTERNAL, PERMS_ERROR_INVALID_INPUT};
use runmat_value::Value;

use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;

use super::{containers, error};

pub(super) async fn apply(value: Value) -> BuiltinResult<Value> {
    if let Value::GpuTensor(handle) = value {
        let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone()))
            .await
            .map_err(internal)?;
        let output = apply_host(gathered)?;
        let restored = gpu_helpers::restore_class_preserving_value(&handle, output, "perms")
            .map_err(internal)?;
        if runmat_accelerate_api::handle_is_explicit(&handle)
            && !matches!(restored, Value::GpuTensor(_))
        {
            return Err(error::with_message(
                &PERMS_ERROR_INTERNAL,
                "perms: provider cannot preserve explicit gpuArray output residency",
            ));
        }
        return Ok(restored);
    }
    apply_host(value)
}

fn apply_host(value: Value) -> BuiltinResult<Value> {
    match value {
        scalar @ (Value::Num(_)
        | Value::Int(_)
        | Value::Complex(_, _)
        | Value::Bool(_)
        | Value::String(_)) => Ok(scalar),
        Value::Tensor(tensor) => containers::numeric(tensor),
        Value::ComplexTensor(tensor) => containers::complex(tensor),
        Value::LogicalArray(array) => containers::logical(array),
        Value::CharArray(array) => containers::characters(array),
        Value::StringArray(array) => containers::strings(array),
        Value::Cell(array) => containers::cells(array),
        _ => Err(error::from_descriptor(&PERMS_ERROR_INVALID_INPUT)),
    }
}

fn internal(detail: impl std::fmt::Display) -> crate::RuntimeError {
    error::with_message(&PERMS_ERROR_INTERNAL, format!("perms: {detail}"))
}
