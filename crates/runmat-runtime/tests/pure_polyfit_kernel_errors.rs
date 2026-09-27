//! Pure solver diagnostics survive a real owning provider round trip.
use runmat_accelerate_api::{
    AccelProvider, GpuTensorStorage, HostNumericDataView, HostNumericTensorView,
};
use runmat_runtime::builtins::math::poly::polyfit::{
    kernel_error::PolyfitKernelError, polyfit_host_real_for_provider,
};
use runmat_value::{Tensor, Value};
use std::error::Error;
use std::sync::atomic::{AtomicUsize, Ordering};

// InProcess intentionally has no polyfit override. This fixture adds exactly
// that hook over genuine InProcess allocations and the production pure solver.
struct SolverProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    calls: AtomicUsize,
}
impl AccelProvider for SolverProvider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        "pure polyfit owning adapter".into()
    }
    fn upload(
        &self,
        view: &runmat_accelerate_api::HostTensorView,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        self.inner.upload(view)
    }
    fn upload_numeric_raw(
        &self,
        view: &HostNumericTensorView,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        self.inner.upload_numeric_raw(view)
    }
    fn download<'a>(
        &'a self,
        _: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        panic!("operational solver errors must not gather or replay")
    }
    fn polyfit<'a>(
        &'a self,
        x: &'a runmat_accelerate_api::GpuTensorHandle,
        y: &'a runmat_accelerate_api::GpuTensorHandle,
        degree: usize,
        weights: Option<&'a runmat_accelerate_api::GpuTensorHandle>,
    ) -> runmat_accelerate_api::AccelProviderFuture<'a, runmat_accelerate_api::ProviderPolyfitResult>
    {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let x = self.inner.download(x).await?;
            let y = self.inner.download(y).await?;
            let weights = match weights {
                Some(handle) => Some(self.inner.download(handle).await?),
                None => None,
            };
            let solved = polyfit_host_real_for_provider(
                &x.data,
                &y.data,
                degree,
                weights.as_ref().map(|w| w.data.as_slice()),
            )?;
            Ok(runmat_accelerate_api::ProviderPolyfitResult {
                coefficients: solved.coefficients,
                r_matrix: solved.r_matrix,
                normr: solved.normr,
                df: solved.df,
                mu: solved.mu,
            })
        })
    }
}

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn pure_solver_preserves_coefficients_and_typed_validation() {
    fn native_error<T: Error + Send + Sync + 'static>() {}
    native_error::<PolyfitKernelError>();
    let result = polyfit_host_real_for_provider(&[0., 1., 2.], &[1., 3., 5.], 1, None).unwrap();
    assert!((result.coefficients[0] - 2.).abs() < 1e-12);
    assert!((result.coefficients[1] - 1.).abs() < 1e-12);
    assert_eq!(result.r_matrix.len(), 4);
    assert_eq!(result.df, 1.);
    assert!(result.normr < 1e-12);
    let error = polyfit_host_real_for_provider(&[0., 1.], &[1.], 1, None)
        .err()
        .unwrap();
    let runtime: runmat_runtime::RuntimeError = error.into();
    assert_eq!(runtime.identifier(), Some("RunMat:polyfit:InvalidInput"));
    assert_eq!(
        runtime.message(),
        "polyfit: X and Y vectors must be the same length"
    );
    assert!(runtime.source().unwrap().is::<PolyfitKernelError>());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn public_weight_errors_match_host_and_retain_provider_cause_and_inputs() {
    let provider = Box::leak(Box::new(SolverProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        calls: AtomicUsize::new(0),
    }));
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for (case, weights) in [[1., -1., 1.], [1., f64::NAN, 1.]].into_iter().enumerate() {
        let data = [[0., 1., 2.], [1., 3., 5.], weights];
        let inputs = data
            .iter()
            .map(|values| {
                provider
                    .upload_numeric(&HostNumericTensorView {
                        data: HostNumericDataView::F64(values),
                        shape: &[3, 1],
                        storage: GpuTensorStorage::Real,
                    })
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let host_args = [
            Value::Tensor(Tensor::new(data[0].to_vec(), vec![3, 1]).unwrap()),
            Value::Tensor(Tensor::new(data[1].to_vec(), vec![3, 1]).unwrap()),
            Value::Num(1.),
            Value::Tensor(Tensor::new(data[2].to_vec(), vec![3, 1]).unwrap()),
        ];
        let resident_args = [
            Value::GpuTensor(inputs[0].clone()),
            Value::GpuTensor(inputs[1].clone()),
            Value::Num(1.),
            Value::GpuTensor(inputs[2].clone()),
        ];
        let host =
            futures::executor::block_on(runmat_runtime::call_builtin_async("polyfit", &host_args))
                .unwrap_err();
        let resident = futures::executor::block_on(runmat_runtime::call_builtin_async(
            "polyfit",
            &resident_args,
        ))
        .unwrap_err();
        assert_eq!(host.identifier(), Some("RunMat:polyfit:InvalidArgument"));
        assert_eq!(resident.identifier(), host.identifier());
        assert_eq!(resident.message(), host.message());
        assert_eq!(
            resident.gpu_gather_retry,
            runmat_runtime::GpuGatherRetry::Never
        );
        let mut cursor: &(dyn Error + 'static) = &resident;
        let mut provider_seen = false;
        loop {
            provider_seen |= cursor.is::<runmat_accelerate_api::ProviderExecutionError>();
            if cursor.is::<PolyfitKernelError>() {
                break;
            }
            cursor = cursor.source().expect("exact solver cause retained");
        }
        assert!(provider_seen, "must exercise the resident provider path");
        assert_eq!(provider.calls.load(Ordering::SeqCst), case + 1);
        for input in &inputs {
            input.allocation_lease().validate_live().unwrap();
        }
    }
}
