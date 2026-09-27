//! Pure solver diagnostics survive a real owning provider round trip.
use runmat_accelerate_api::{
    AccelProvider, GpuTensorStorage, HostNumericDataView, HostNumericTensorView,
};
use runmat_runtime::builtins::math::poly::polyfit::{
    kernel_error::PolyfitKernelError, polyfit_host_real_for_provider,
};
use runmat_value::{Tensor, Value};
use std::error::Error;

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
    let provider = Box::leak(Box::new(
        runmat_accelerate::simple_provider::InProcessProvider::new(),
    ));
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for weights in [[1., -1., 1.], [1., f64::NAN, 1.]] {
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
        for input in &inputs {
            input.allocation_lease().validate_live().unwrap();
        }
    }
}
