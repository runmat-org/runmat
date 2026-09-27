use runmat_accelerate_api::*;
use runmat_runtime::builtins::math::reduction::gradient::{
    gradient_complex_tensor_host, gradient_real_tensor_host,
    gradient_real_tensor_host_with_coordinates, kernel_error::GradientKernelError,
};
use runmat_value::{ComplexTensor, NumericDType, Tensor, Value};
use std::error::Error;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

fn malformed_storage_error() -> GradientKernelError {
    // Deliberate host metadata corruption exercises the kernel's real storage
    // validation. No fabricated GPU handle or allocation identity is involved.
    let mut tensor = Tensor::new(vec![1., 2., 3.], vec![1, 3]).unwrap();
    tensor.shape = vec![1, 2];
    gradient_real_tensor_host(tensor, 2, 1.).unwrap_err()
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
        "gradient metadata fault fixture".into()
    }
    fn upload(&self, view: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(view)
    }
    fn upload_numeric_raw(&self, view: &HostNumericTensorView) -> anyhow::Result<GpuTensorHandle> {
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
    ) -> anyhow::Result<GpuTensorHandle> {
        input.allocation_lease().validate_live().unwrap();
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(anyhow::Error::new(malformed_storage_error()))
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn pure_native_gradient_values_and_error_metadata_are_preserved() {
    fn native_error<T: Error + Send + Sync + 'static>() {}
    native_error::<GradientKernelError>();
    let input = Tensor::from_f32(vec![0., 2., 8.], vec![1, 3]).unwrap();
    let output = gradient_real_tensor_host(input.clone(), 2, 2.).unwrap();
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    assert_eq!(output.shape, [1, 3]);
    assert_eq!(output.materialize_f64(), [1., 2., 3.]);
    let coordinates =
        gradient_real_tensor_host_with_coordinates(input, 2, vec![0., 1., 4.]).unwrap();
    assert_eq!(coordinates.materialize_f64(), [2., 2., 2.]);
    let complex = ComplexTensor::from_f64_values_with_dtype(
        vec![(0., 0.), (2., 4.), (8., 16.)],
        vec![1, 3],
        NumericDType::F32,
    )
    .unwrap();
    let output = gradient_complex_tensor_host(complex, 2, 2.).unwrap();
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    assert_eq!(output.materialize_f64(), [(1., 2.), (2., 4.), (3., 6.)]);
    let error: runmat_runtime::RuntimeError = malformed_storage_error().into();
    assert_eq!(error.identifier(), Some("RunMat:gradient:Internal"));
    assert!(error.source().unwrap().is::<GradientKernelError>());
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
    let expected: runmat_runtime::RuntimeError = malformed_storage_error().into();
    let error = futures::executor::block_on(runmat_runtime::call_builtin_async(
        "gradient",
        &[Value::GpuTensor(input.clone())],
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), expected.identifier());
    assert_eq!(error.message(), expected.message());
    assert_eq!(
        error.gpu_gather_retry,
        runmat_runtime::GpuGatherRetry::Never
    );
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
