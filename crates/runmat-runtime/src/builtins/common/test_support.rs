use runmat_builtins::{LogicalArray, Tensor, Value};

/// Extract MATLAB code blocks from builtin documentation.
pub fn doc_examples(doc: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut current = Vec::new();
    let mut in_block = false;

    for line in doc.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```matlab") {
            in_block = true;
            current.clear();
            continue;
        }
        if trimmed.starts_with("```") && in_block {
            blocks.push(current.join("\n"));
            in_block = false;
            continue;
        }
        if in_block {
            current.push(line.trim_end().to_string());
        }
    }

    if in_block && !current.is_empty() {
        blocks.push(current.join("\n"));
    }

    blocks
}

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
