#[cfg(feature = "wgpu")]
use futures::executor::block_on;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, AccelProviderFuture, GpuTensorHandle, GpuTensorStorage,
    HostTensorOwned, HostTensorView, NumericElementType,
};
use runmat_value::{Tensor, Value};

use crate::builtins::common::test_support;

use super::execute;

struct FallbackProvider;

impl AccelProvider for FallbackProvider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        Ok(GpuTensorHandle {
            shape: host.shape.to_vec(),
            device_id: self.device_id(),
            buffer_id: 2,
            descriptor: Default::default(),
        }
        .with_numeric_descriptor(NumericElementType::F64, GpuTensorStorage::Real))
    }

    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            Ok(HostTensorOwned {
                data: if handle.buffer_id == 2 {
                    vec![0.0, 0.5, 1.0]
                } else {
                    vec![-1.0, 0.0, 2.0]
                },
                shape: vec![1, 3],
                storage: GpuTensorStorage::Real,
            })
        })
    }

    fn free(&self, _: &GpuTensorHandle) -> anyhow::Result<()> {
        Ok(())
    }

    fn device_info(&self) -> String {
        "heaviside-fallback-test".to_owned()
    }

    fn device_id(&self) -> u32 {
        17
    }

    fn unary_heaviside<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async {
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "unary_heaviside unavailable",
            ))
        })
    }
}

struct FailingProvider;

impl AccelProvider for FailingProvider {
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        unreachable!("terminal provider failure must not enter fallback")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { Err(anyhow::anyhow!("download must not run")) })
    }

    fn free(&self, _: &GpuTensorHandle) -> anyhow::Result<()> {
        Ok(())
    }

    fn device_info(&self) -> String {
        "heaviside-failure-test".to_owned()
    }

    fn device_id(&self) -> u32 {
        19
    }

    fn unary_heaviside<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async { Err(anyhow::anyhow!("device lost")) })
    }
}

fn handle(provider: &dyn AccelProvider) -> GpuTensorHandle {
    GpuTensorHandle {
        shape: vec![1, 3],
        device_id: provider.device_id(),
        buffer_id: 1,
        descriptor: Default::default(),
    }
    .with_numeric_descriptor(NumericElementType::F64, GpuTensorStorage::Real)
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
                shape: &tensor.shape,
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
    let provider: &'static dyn AccelProvider = Box::leak(Box::new(FallbackProvider));
    let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let input = handle(provider);
    let result = execute(Value::GpuTensor(input.clone())).unwrap();
    let Value::GpuTensor(output) = &result else {
        panic!("expected resident fallback result")
    };
    assert_eq!(output.device_id, input.device_id);
    assert_ne!(output.buffer_id, input.buffer_id);
    assert!(std::ptr::eq(
        runmat_accelerate_api::provider_for_handle(output).unwrap(),
        provider
    ));
    assert_eq!(
        test_support::gather(result).unwrap().materialize_f64(),
        vec![0.0, 0.5, 1.0]
    );
}

#[test]
fn terminal_provider_error_does_not_fallback() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = test_support::accel_test_lock();
    let provider: &'static dyn AccelProvider = Box::leak(Box::new(FailingProvider));
    let _active = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let error = execute(Value::GpuTensor(handle(provider))).expect_err("terminal failure");
    assert_eq!(error.identifier(), Some("RunMat:heaviside:ProviderFailed"));
    assert!(error.message().contains("device lost"));
    assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
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
            shape: &tensor.shape,
        })
        .unwrap();
    let result = block_on(super::super::provider::execute(input)).unwrap();
    let actual = test_support::gather(result).unwrap();
    assert_eq!(actual.shape, expected.shape);
    for (actual, expected) in actual
        .materialize_f64()
        .iter()
        .zip(expected.materialize_f64())
    {
        assert!(actual == &expected || actual.is_nan() && expected.is_nan());
    }
}
