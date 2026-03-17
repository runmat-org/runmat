use crate::build_runtime_error;
use futures::executor::block_on;
use runmat_builtins::{LogicalArray, Tensor, Value};

pub mod fs {
    use std::io;
    use std::path::Path;

    pub fn write(path: impl AsRef<Path>, data: impl AsRef<[u8]>) -> io::Result<()> {
        futures::executor::block_on(runmat_filesystem::write_async(path, data))
    }

    pub fn remove_file(path: impl AsRef<Path>) -> io::Result<()> {
        futures::executor::block_on(runmat_filesystem::remove_file_async(path))
    }

    pub fn read(path: impl AsRef<Path>) -> io::Result<Vec<u8>> {
        futures::executor::block_on(runmat_filesystem::read_async(path))
    }

    pub fn read_to_string(path: impl AsRef<Path>) -> io::Result<String> {
        futures::executor::block_on(runmat_filesystem::read_to_string_async(path))
    }

    pub fn create_dir(path: impl AsRef<Path>) -> io::Result<()> {
        futures::executor::block_on(runmat_filesystem::create_dir_async(path))
    }

    pub fn create_dir_all(path: impl AsRef<Path>) -> io::Result<()> {
        futures::executor::block_on(runmat_filesystem::create_dir_all_async(path))
    }
}

/// Ensure an in-process acceleration provider is registered for tests,
/// invoking the supplied closure with the provider trait object.
pub fn with_test_provider<F, R>(f: F) -> R
where
    F: FnOnce(&'static dyn runmat_accelerate_api::AccelProvider) -> R,
{
    for _ in 0..5 {
        runmat_accelerate::simple_provider::register_inprocess_provider();
        runmat_accelerate::simple_provider::reset_inprocess_rng();
        if let Some(provider) = runmat_accelerate_api::provider() {
            let _guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
            return f(provider);
        }
        std::thread::yield_now();
    }
    panic!("test provider registered");
}

/// Gather a value (recursively) so assertions can operate on host tensors.
pub fn gather(value: Value) -> Result<Tensor, crate::RuntimeError> {
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
    #[cfg(not(target_arch = "wasm32"))]
    let provider = match &value {
        Value::GpuTensor(handle) => runmat_accelerate_api::provider_for_handle(handle),
        _ => runmat_accelerate_api::provider(),
    };

    #[cfg(not(target_arch = "wasm32"))]
    let gathered = {
        let _guard = runmat_accelerate_api::ThreadProviderGuard::set(provider);
        block_on(crate::dispatcher::gather_if_needed_async(&value))?
    };

    #[cfg(target_arch = "wasm32")]
    let gathered = block_on(crate::dispatcher::gather_if_needed_async(&value))?;

    match gathered {
        Value::Tensor(t) => Ok(t),
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1])
            .map_err(|e| build_runtime_error(format!("gather: {e}")).build()),
        Value::LogicalArray(LogicalArray { data, shape }) => {
            let dense: Vec<f64> = data
                .iter()
                .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                .collect();
            Tensor::new(dense, shape.clone())
                .map_err(|e| build_runtime_error(format!("gather: {e}")).build())
        }
        other => Err(build_runtime_error(format!("gather: unsupported value {other:?}")).build()),
    }
}
