use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{Tensor, Value};

fn encode_internal_suspend(prompt: &str) -> String {
    // Transitional marker to carry a typed suspension through legacy `Result<_, String>` call
    // stacks. `dispatcher::call_builtin` recognizes this and converts it back into a typed
    // suspension.
    format!("__RUNMAT_SUSPEND__:internal:{prompt}")
}

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
    let gathered = match crate::dispatcher::gather_if_needed(&value) {
        Ok(v) => v,
        Err(runmat_async::RuntimeControlFlow::Error(e)) => return Err(e),
        Err(runmat_async::RuntimeControlFlow::Suspend(pending)) => {
            // Preserve suspension through string-based call stacks.
            return Err(encode_internal_suspend(&pending.prompt));
        }
    };
    match gathered {
        Value::Tensor(t) => Ok(t),
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("gather: {e}").into()),
        Value::LogicalArray(la) => {
            let data: Vec<f64> = la
                .data
                .iter()
                .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                .collect();
            Tensor::new(data, la.shape.clone()).map_err(|e| format!("gather: {e}").into())
        }
        other => Err(format!("gather: unexpected value kind {other:?}").into()),
    }
}

/// Gather an arbitrary value, returning a host-side `Value`.
pub fn gather_value(value: &Value) -> Result<Value, String> {
    match crate::dispatcher::gather_if_needed(value) {
        Ok(v) => Ok(v),
        Err(runmat_async::RuntimeControlFlow::Error(e)) => Err(e),
        Err(runmat_async::RuntimeControlFlow::Suspend(pending)) => {
            Err(encode_internal_suspend(&pending.prompt))
        }
    }
}

/// Wrap a GPU tensor handle as a logical gpuArray value, recording metadata so that
/// predicates like `islogical` can inspect the handle without downloading it.
pub fn logical_gpu_value(handle: GpuTensorHandle) -> Value {
    runmat_accelerate_api::set_handle_logical(&handle, true);
    Value::GpuTensor(handle)
}
