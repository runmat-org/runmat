use super::*;
use crate::builtins::common::test_support;
#[test]
fn resident_like_logical_upload_is_owned_and_preserves_nd_empty_shapes() {
    test_support::with_test_provider(|provider| {
        let prototype = provider
            .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                data: &[1],
                shape: &[1, 1],
            })
            .unwrap();
        for (data, shape) in [
            (vec![0.0, -2.0, 3.0, 0.0], vec![2, 1, 2]),
            (vec![], vec![0, 3, 2]),
        ] {
            let expected = data
                .iter()
                .map(|value| if *value == 0.0 { 0.0 } else { 1.0 })
                .collect::<Vec<_>>();
            let Value::GpuTensor(output) =
                tensor_to_gpu_value(Tensor::new(data, shape.clone()).unwrap(), &prototype).unwrap()
            else {
                panic!("resident Like result");
            };
            assert!(output.view().is_logical());
            assert!(!output.shares_allocation_with(&prototype));
            assert_eq!(output.shape(), shape);
            assert_eq!(
                futures::executor::block_on(provider.download(&output))
                    .unwrap()
                    .data,
                expected
            );
        }
        assert!(futures::executor::block_on(provider.download(&prototype)).is_ok());
    });
}
use crate::builtins::io::filetext::registry;
use crate::builtins::io::filetext::{fclose, fopen};
use crate::RuntimeError;
use runmat_accelerate_api::AccelProvider as _;
use runmat_filesystem::File;
#[cfg(not(target_arch = "wasm32"))]
use runmat_time::system_time_now;
use std::io::Write;
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::time::UNIX_EPOCH;

struct FreadOwnerProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    precision: runmat_accelerate_api::ProviderPrecision,
}

impl FreadOwnerProvider {
    fn new(precision: runmat_accelerate_api::ProviderPrecision) -> Self {
        Self {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            precision,
        }
    }
}

impl runmat_accelerate_api::AccelProvider for FreadOwnerProvider {
    fn upload(
        &self,
        host: &runmat_accelerate_api::HostTensorView,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        self.inner.upload(host)
    }

    fn download<'a>(
        &'a self,
        handle: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }

    fn upload_numeric_raw(
        &self,
        host: &runmat_accelerate_api::HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<runmat_accelerate_api::GpuTensorHandle>
    {
        self.inner.upload_numeric_raw(host)
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
        self.inner.download_numeric(handle)
    }

    fn upload_integer(
        &self,
        host: &runmat_accelerate_api::HostIntegerTensorView,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        self.inner.upload_integer(host)
    }

    fn download_integer<'a>(
        &'a self,
        handle: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelIntegerDownloadFuture<'a> {
        self.inner.download_integer(handle)
    }

    fn device_info(&self) -> String {
        self.inner.device_info()
    }

    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }

    fn precision(&self) -> runmat_accelerate_api::ProviderPrecision {
        self.precision
    }
}
fn unwrap_error_message(err: RuntimeError) -> String {
    err.message().to_string()
}

fn run_evaluate(fid_value: &Value, rest: &[Value]) -> BuiltinResult<FreadEval> {
    futures::executor::block_on(evaluate(fid_value, rest))
}

fn run_fopen(args: &[Value]) -> BuiltinResult<fopen::FopenEval> {
    futures::executor::block_on(fopen::evaluate(args))
}

fn run_fclose(args: &[Value]) -> BuiltinResult<fclose::FcloseEval> {
    futures::executor::block_on(fclose::evaluate(args))
}

fn double_values(tensor: &Tensor) -> &[f64] {
    tensor.as_f64_slice().expect("expected double tensor")
}

#[cfg(feature = "wgpu")]
fn run_call_builtin(name: &str, args: &[Value]) -> BuiltinResult<crate::execution::RootedValue> {
    crate::call_builtin(name, args)
}

fn registry_guard() -> registry::RegistryTestGuard {
    registry::test_guard()
}

#[cfg(not(target_arch = "wasm32"))]
fn unique_path(prefix: &str) -> PathBuf {
    let now = system_time_now()
        .duration_since(UNIX_EPOCH)
        .expect("time went backwards");
    let filename = format!(
        "runmat_{prefix}_{}_{}.tmp",
        now.as_secs(),
        now.subsec_nanos()
    );
    std::env::temp_dir().join(filename)
}

#[cfg(target_arch = "wasm32")]
fn unique_path(prefix: &str) -> PathBuf {
    test_support::filesystem::unique_path(prefix)
}
mod resident;
mod admission;
mod controls;
mod decoding;
mod io;
