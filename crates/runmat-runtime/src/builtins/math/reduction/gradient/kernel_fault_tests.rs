use super::GradientKernelError;
use runmat_accelerate_api::*;
use runmat_value::Value;
use std::error::Error;
use std::sync::atomic::{AtomicUsize, Ordering};

// Inject a typed provider execution fault without violating Tensor invariants.
fn kernel_fault() -> GradientKernelError {
    GradientKernelError::new("gradient: injected provider execution fault")
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn native_kernel_fault_retains_identifier_and_typed_source() {
    let error: crate::RuntimeError = kernel_fault().into();
    assert_eq!(error.identifier(), Some("RunMat:gradient:Internal"));
    assert!(error.source().unwrap().is::<GradientKernelError>());
}

struct FailingGradient {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    calls: AtomicUsize,
}
impl AccelProvider for FailingGradient {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        "gradient execution fault fixture".into()
    }
    fn upload(&self, view: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(view)
    }
    fn upload_numeric_raw(
        &self,
        view: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.inner.upload_numeric_raw(view)
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("kernel execution failure must not gather or replay")
    }
    fn gradient_dim(
        &self,
        input: &GpuTensorHandle,
        _: usize,
        _: f64,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let result: anyhow::Result<_> = {
            input.allocation_lease().validate_live().unwrap();
            self.calls.fetch_add(1, Ordering::SeqCst);
            Err(anyhow::Error::new(kernel_fault()))
        };
        result.map(Some).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
        })
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn public_gradient_retains_kernel_fault_without_retry_or_owner_loss() {
    let provider = Box::leak(Box::new(FailingGradient {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        calls: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[1., 2., 3.]),
            shape: &[1, 3],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let expected: crate::RuntimeError = kernel_fault().into();
    let error = futures::executor::block_on(crate::call_builtin_async(
        "gradient",
        &[Value::GpuTensor(input.clone())],
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), expected.identifier());
    assert_eq!(error.message(), expected.message());
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    let mut cursor: &(dyn Error + 'static) = &error;
    let mut provider_seen = false;
    while !cursor.is::<GradientKernelError>() {
        provider_seen |= cursor.is::<ProviderExecutionError>();
        cursor = cursor
            .source()
            .expect("exact kernel source survives dispatch");
    }
    assert!(provider_seen);
    input.allocation_lease().validate_live().unwrap();
}
