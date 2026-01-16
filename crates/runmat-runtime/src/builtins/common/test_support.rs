use runmat_builtins::{LogicalArray, Tensor, Value};

/// Ensure an in-process acceleration provider is registered for tests,
/// invoking the supplied closure with the provider trait object.
pub fn with_test_provider<F, R>(f: F) -> R
where
    F: FnOnce(&'static dyn runmat_accelerate_api::AccelProvider) -> R,
{
    runmat_accelerate::simple_provider::register_inprocess_provider();
    runmat_accelerate::simple_provider::reset_inprocess_rng();
    let provider = runmat_accelerate_api::provider().expect("test provider registered");
    let _guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    f(provider)
}

/// Gather a value (recursively) so assertions can operate on host tensors.
pub fn gather(value: Value) -> Result<Tensor, String> {
    // Ensure the correct provider is active for GPU handles created by the WGPU backend.
    #[cfg(feature = "wgpu")]
    {
        if let Value::GpuTensor(ref h) = value {
            if h.device_id != 0 {
                let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                    runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
                );
            }
        }
    }
    match crate::dispatcher::gather_if_needed(&value)? {
        Value::Tensor(t) => Ok(t),
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).map_err(|e| format!("gather: {e}")),
        Value::LogicalArray(LogicalArray { data, shape }) => {
            let dense: Vec<f64> = data
                .iter()
                .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                .collect();
            Tensor::new(dense, shape.clone()).map_err(|e| format!("gather: {e}"))
        }
        other => Err(format!("gather: unsupported value {other:?}")),
    }
}
