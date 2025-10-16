use runmat_builtins::{Tensor, Value};

/// Download a GPU tensor handle to host memory, returning a dense `Tensor`.
///
/// This helper routes through the dispatcher so residency hooks and provider
/// semantics stay consistent with the rest of the runtime.
pub fn gather_tensor(handle: &runmat_accelerate_api::GpuTensorHandle) -> Result<Tensor, String> {
    let value = Value::GpuTensor(handle.clone());
    match crate::dispatcher::gather_if_needed(&value)? {
        Value::Tensor(t) => Ok(t),
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("gather: {e}")),
        other => Err(format!("gather: unexpected value kind {other:?}")),
    }
}

/// Gather an arbitrary value, returning a host-side `Value`.
pub fn gather_value(value: &Value) -> Result<Value, String> {
    crate::dispatcher::gather_if_needed(value)
}
