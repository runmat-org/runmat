use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{Tensor, Value};

/// Download a GPU tensor handle to host memory, returning a dense `Tensor`.
///
/// This helper routes through the dispatcher so residency hooks and provider
/// semantics stay consistent with the rest of the runtime.
pub fn gather_tensor(handle: &runmat_accelerate_api::GpuTensorHandle) -> Result<Tensor, String> {
    // Ensure the correct provider is active for WGPU-backed handles when tests run in parallel.
    // This mirrors the guard used in test_support::gather.
    #[cfg(all(test, feature = "wgpu"))]
    {
        if handle.device_id != 0 {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    let value = Value::GpuTensor(handle.clone());
    match crate::dispatcher::gather_if_needed(&value)? {
        Value::Tensor(t) => Ok(t),
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("gather: {e}")),
        Value::LogicalArray(la) => {
            let data: Vec<f64> = la
                .data
                .iter()
                .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                .collect();
            Tensor::new(data, la.shape.clone()).map_err(|e| format!("gather: {e}"))
        }
        other => Err(format!("gather: unexpected value kind {other:?}")),
    }
}

/// Gather an arbitrary value, returning a host-side `Value`.
pub fn gather_value(value: &Value) -> Result<Value, String> {
    crate::dispatcher::gather_if_needed(value)
}

/// Wrap a GPU tensor handle as a logical gpuArray value, recording metadata so that
/// predicates like `islogical` can inspect the handle without downloading it.
pub fn logical_gpu_value(handle: GpuTensorHandle) -> Value {
    runmat_accelerate_api::set_handle_logical(&handle, true);
    Value::GpuTensor(handle)
}
