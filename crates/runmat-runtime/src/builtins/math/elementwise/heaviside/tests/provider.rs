#[cfg(feature = "wgpu")]
use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, AccelProviderFuture,
    GpuTensorHandle, HostNumericTensorView, HostTensorView,
};
use runmat_value::{Tensor, Value};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::builtins::common::test_support;

use super::execute;

struct FallbackProvider {
    inner: InProcessProvider,
    terminal: bool,
    transfers: AtomicUsize,
}

impl FallbackProvider {
    fn new(terminal: bool) -> Self {
        Self {
            inner: InProcessProvider::new(),
            terminal,
            transfers: AtomicUsize::new(0),
        }
    }
    fn input(&self) -> GpuTensorHandle {
        self.inner
            .upload(&HostTensorView {
                data: &[-1.0, 0.0, 2.0],
                shape: &[1, 3],
            })
            .expect("actual input allocation")
    }
}

impl AccelProvider for FallbackProvider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.transfers.fetch_add(1, Ordering::SeqCst);
        self.inner.upload(host)
    }

    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.transfers.fetch_add(1, Ordering::SeqCst);
        self.inner.download(handle)
    }

    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.transfers.fetch_add(1, Ordering::SeqCst);
        self.inner.upload_numeric_raw(host)
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.transfers.fetch_add(1, Ordering::SeqCst);
        self.inner.download_numeric(handle)
    }

    fn device_info(&self) -> String {
        "heaviside-fallback-test".to_owned()
    }

    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }

    fn unary_heaviside<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.terminal {
                return Err(anyhow::anyhow!("device lost"));
            }
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "unary_heaviside unavailable",
            ))
        })
    }
}

#[test]
fn direct_provider_roundtrip_stays_resident() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![-3.0, -0.0, 0.0, 2.5], vec![2, 2]).unwrap();
        let values = tensor.materialize_f64();
        let input = provider
            .upload(&HostTensorView {
                data: &values,
                shape: tensor.shape(),
            })
            .unwrap();
        let result = execute(Value::GpuTensor(input)).unwrap();
        assert!(matches!(result, Value::GpuTensor(_)));
        assert_eq!(
            test_support::gather(result).unwrap().materialize_f64(),
            vec![0.0, 0.5, 0.5, 1.0]
        );
    });
}

#[test]
fn typed_unsupported_hook_uses_exact_owner_fallback() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = test_support::accel_test_lock();
    let provider = Box::leak(Box::new(FallbackProvider::new(false)));
    let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let input = provider.input();
    let input_lease = input.allocation_lease().downgrade();
    let result = execute(Value::GpuTensor(input.clone())).unwrap();
    let Value::GpuTensor(output) = &result else {
        panic!("expected resident fallback result")
    };
    assert_eq!(output.device_id(), input.device_id());
    assert!(!output.shares_allocation_with(&input));
    let output_lease = output.allocation_lease().downgrade();
    assert!(runmat_accelerate_api::same_provider_instance(
        runmat_accelerate_api::provider_for_handle(output).unwrap(),
        provider
    ));
    assert_eq!(
        test_support::gather(result).unwrap().materialize_f64(),
        vec![0.0, 0.5, 1.0]
    );
    assert!(output_lease.acquire().is_none());
    assert!(input_lease.acquire().is_some());
    drop(input);
    assert!(input_lease.acquire().is_none());
}

#[test]
fn terminal_provider_error_does_not_fallback() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = test_support::accel_test_lock();
    let provider = Box::leak(Box::new(FallbackProvider::new(true)));
    let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let input = provider.input();
    let lease = input.allocation_lease().downgrade();
    let error = execute(Value::GpuTensor(input.clone())).expect_err("terminal failure");
    assert_eq!(error.identifier(), Some("RunMat:heaviside:ProviderFailed"));
    assert!(error.message().contains("device lost"));
    assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
    assert_eq!(provider.transfers.load(Ordering::SeqCst), 0);
    assert!(lease.acquire().is_some());
    drop((error, input));
    assert!(lease.acquire().is_none());
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_matches_host() {
    let _guard = test_support::accel_test_lock();
    let Some(provider) = test_support::wgpu_provider_if_available() else {
        return;
    };
    let tensor = Tensor::new(vec![-3.0, -0.0, 0.0, 4.0, f64::NAN], vec![1, 5]).unwrap();
    let expected = super::super::host::apply(tensor.clone()).unwrap();
    let values = tensor.materialize_f64();
    let input = provider
        .upload(&HostTensorView {
            data: &values,
            shape: tensor.shape(),
        })
        .unwrap();
    let result = block_on(super::super::provider::execute(input)).unwrap();
    let actual = test_support::gather(result).unwrap();
    assert_eq!(actual.shape(), expected.shape());
    for (actual, expected) in actual
        .materialize_f64()
        .iter()
        .zip(expected.materialize_f64())
    {
        assert!(actual == &expected || actual.is_nan() && expected.is_nan());
    }
}
