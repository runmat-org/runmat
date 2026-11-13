//! Pure solver diagnostics survive a real owning provider round trip.
use runmat_accelerate_api::{
    AccelProvider, GpuTensorStorage, HostNumericDataView, HostNumericTensorView,
    ProviderPolyfitOutputs,
};
use runmat_runtime::builtins::math::poly::polyfit::{
    kernel_error::PolyfitKernelError, polyfit_host_real_for_provider,
};
use runmat_value::{Tensor, Value};
use std::error::Error;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

// InProcess intentionally has no polyfit override. This fixture adds exactly
// that hook over genuine InProcess allocations and the production pure solver.
struct SolverProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    calls: AtomicUsize,
    override_result: Mutex<Option<runmat_accelerate_api::ProviderPolyfitResult>>,
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
    ) -> runmat_accelerate_api::ProviderExecutionResult<runmat_accelerate_api::GpuTensorHandle>
    {
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
        outputs: ProviderPolyfitOutputs,
    ) -> runmat_accelerate_api::AccelProviderFuture<'a, runmat_accelerate_api::ProviderPolyfitResult>
    {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if let Some(result) = self.override_result.lock().unwrap().clone() {
                return Ok(result);
            }
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
                outputs,
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
    let result = polyfit_host_real_for_provider(
        &[0., 1., 2.],
        &[1., 3., 5.],
        1,
        None,
        ProviderPolyfitOutputs::Statistics,
    )
    .unwrap();
    assert!((result.coefficients[0] - 2.).abs() < 1e-12);
    assert!((result.coefficients[1] - 1.).abs() < 1e-12);
    assert_eq!(result.r_matrix.as_ref().unwrap().len(), 4);
    assert_eq!(result.df, 1.);
    assert!(result.normr.unwrap() < 1e-12);
    let error = polyfit_host_real_for_provider(
        &[0., 1.],
        &[1.],
        1,
        None,
        ProviderPolyfitOutputs::Statistics,
    )
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
fn pure_provider_solver_uses_requested_coefficient_and_qr_basis() {
    let x = [10.0, 11.0, 12.0, 13.0];
    let y = [21.0, 24.0, 24.0, 28.0];
    let original =
        polyfit_host_real_for_provider(&x, &y, 1, None, ProviderPolyfitOutputs::Statistics)
            .unwrap();
    let centered = polyfit_host_real_for_provider(
        &x,
        &y,
        1,
        None,
        ProviderPolyfitOutputs::StatisticsAndNormalization,
    )
    .unwrap();
    assert!((original.coefficients[0] - 2.1).abs() < 1e-11);
    assert!((original.coefficients[1] - 0.1).abs() < 1e-10);
    assert_eq!(original.mu, [0.0, 1.0]);
    assert!((centered.coefficients[0] - 2.1 * centered.mu[1]).abs() < 1e-11);
    assert!((centered.coefficients[1] - 24.25).abs() < 1e-11);
    assert!((centered.mu[0] - 11.5).abs() < 1e-12);
    assert_ne!(original.r_matrix, centered.r_matrix);
    for (index, (original_x, observed)) in x.into_iter().zip(y).enumerate() {
        let original_fit = original
            .coefficients
            .iter()
            .fold(0.0, |acc, coefficient| acc * original_x + coefficient);
        let normalized_x = (original_x - centered.mu[0]) / centered.mu[1];
        let centered_fit = centered
            .coefficients
            .iter()
            .fold(0.0, |acc, coefficient| acc * normalized_x + coefficient);
        assert!(
            (original_fit - centered_fit).abs() < 1e-10,
            "sample {index}"
        );
        assert!((original_fit - observed).abs() < 1.4, "sample {index}");
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn impossible_degree_rejects_before_workspace_or_provider_work() {
    let error = polyfit_host_real_for_provider(
        &[0., 1.],
        &[1., 3.],
        usize::MAX,
        None,
        ProviderPolyfitOutputs::Statistics,
    )
    .err()
    .unwrap();
    assert!(matches!(error, PolyfitKernelError::InvalidArgument(_)));
    let fit = polyfit_host_real_for_provider(
        &[-1., 1.],
        &[2., 4.],
        3,
        None,
        ProviderPolyfitOutputs::Statistics,
    )
    .unwrap();
    assert_eq!(fit.coefficients.len(), 4);
    assert_eq!(fit.r_matrix.as_ref().unwrap().len(), 16);
    assert_eq!(fit.df, 0.);
    for (x, y) in [(-1., 2.), (1., 4.)] {
        let evaluated = fit
            .coefficients
            .iter()
            .fold(0., |value, coefficient| value * x + coefficient);
        assert!((evaluated - y).abs() < 1e-12);
    }
    let provider = Box::leak(Box::new(SolverProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        calls: AtomicUsize::new(0),
        override_result: Mutex::new(None),
    }));
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let x = provider
        .inner
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &[f64::NAN, 1.],
            shape: &[2, 1],
        })
        .unwrap();
    let y = provider
        .inner
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &[1., 3.],
            shape: &[2, 1],
        })
        .unwrap();
    let combined: runmat_runtime::RuntimeError = polyfit_host_real_for_provider(
        &[f64::NAN, 1.],
        &[1., 3.],
        usize::MAX,
        None,
        ProviderPolyfitOutputs::Statistics,
    )
    .err()
    .unwrap()
    .into();
    for resident in [false, true] {
        let args = if resident {
            vec![
                Value::GpuTensor(x.clone()),
                Value::GpuTensor(y.clone()),
                Value::Int(runmat_value::IntValue::U64(usize::MAX as u64)),
            ]
        } else {
            vec![
                Value::Tensor(Tensor::new(vec![f64::NAN, 1.], vec![2, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![1., 3.], vec![2, 1]).unwrap()),
                Value::Int(runmat_value::IntValue::U64(usize::MAX as u64)),
            ]
        };
        let error =
            futures::executor::block_on(runmat_runtime::call_builtin_async("polyfit", &args))
                .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:polyfit:InvalidArgument"));
        assert_eq!(error.message(), combined.message());
        let mut source: &(dyn Error + 'static) = &error;
        while !source.is::<PolyfitKernelError>() {
            source = source.source().expect("typed geometry failure");
        }
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    }
    x.allocation_lease().validate_live().unwrap();
    y.allocation_lease().validate_live().unwrap();
    // Structural length checks precede geometry; numerical weight and sample
    // values are checked only after the requested workspace is addressable.
    let mismatch = polyfit_host_real_for_provider(
        &[0., 1.],
        &[1.],
        usize::MAX,
        None,
        ProviderPolyfitOutputs::Statistics,
    )
    .err()
    .unwrap();
    assert_eq!(
        mismatch.to_string(),
        "polyfit: X and Y vectors must be the same length"
    );
    let weights = polyfit_host_real_for_provider(
        &[0., 1.],
        &[1., 3.],
        usize::MAX,
        Some(&[-1., 1.]),
        ProviderPolyfitOutputs::Statistics,
    )
    .err()
    .unwrap();
    assert_eq!(weights.to_string(), combined.message());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn impossible_geometry_precedes_weight_values_on_all_routes_without_resident_gather() {
    let provider = Box::leak(Box::new(SolverProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        calls: AtomicUsize::new(0),
        override_result: Mutex::new(None),
    }));
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for weights in [[-1., 1.], [f64::NAN, 1.]] {
        let data = [[0., 1.], [1., 3.], weights];
        let expected: runmat_runtime::RuntimeError = polyfit_host_real_for_provider(
            &data[0],
            &data[1],
            usize::MAX,
            Some(&weights),
            ProviderPolyfitOutputs::Statistics,
        )
        .err()
        .unwrap()
        .into();
        let inputs = data
            .iter()
            .map(|values| {
                provider
                    .inner
                    .upload(&runmat_accelerate_api::HostTensorView {
                        data: values,
                        shape: &[2, 1],
                    })
                    .unwrap()
            })
            .collect::<Vec<_>>();
        for resident in [false, true] {
            let mut args = if resident {
                inputs
                    .iter()
                    .cloned()
                    .map(Value::GpuTensor)
                    .collect::<Vec<_>>()
            } else {
                data.iter()
                    .map(|values| Value::Tensor(Tensor::new(values.to_vec(), vec![2, 1]).unwrap()))
                    .collect::<Vec<_>>()
            };
            args.insert(
                2,
                Value::Int(runmat_value::IntValue::U64(usize::MAX as u64)),
            );
            let error =
                futures::executor::block_on(runmat_runtime::call_builtin_async("polyfit", &args))
                    .unwrap_err();
            assert_eq!(error.identifier(), expected.identifier());
            assert_eq!(error.message(), expected.message());
            assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
        }
        for input in &inputs {
            input.allocation_lease().validate_live().unwrap();
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn invalid_host_degree_precedes_resident_integer_download() {
    let provider = Box::leak(Box::new(SolverProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        calls: AtomicUsize::new(0),
        override_result: Mutex::new(None),
    }));
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let integers = [1_i64, 2];
    let x = provider
        .inner
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::I64(&integers),
            shape: &[2, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let y = provider
        .inner
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &[1., 3.],
            shape: &[2, 1],
        })
        .unwrap();
    for degree in [-1., 1.5] {
        let error = futures::executor::block_on(runmat_runtime::call_builtin_async(
            "polyfit",
            &[
                Value::GpuTensor(x.clone()),
                Value::GpuTensor(y.clone()),
                Value::Num(degree),
            ],
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:polyfit:InvalidArgument"));
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    }
    x.allocation_lease().validate_live().unwrap();
    y.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn extreme_original_basis_fits_coefficients_without_unrequested_overflowing_r() {
    let x_data = [f64::MAX, -f64::MAX];
    let y_data = [1.0, -1.0];
    let coefficient_only = polyfit_host_real_for_provider(
        &x_data,
        &y_data,
        1,
        None,
        ProviderPolyfitOutputs::Coefficients,
    )
    .expect("finite coefficients do not require an unrepresentable R matrix");
    assert!(coefficient_only.r_matrix.is_none());
    assert!(coefficient_only.coefficients[0] > 0.0);
    assert!((coefficient_only.coefficients[0] * f64::MAX - 1.0).abs() < 1e-14);
    assert!(coefficient_only.coefficients[1].abs() < 1e-14);
    let error = polyfit_host_real_for_provider(
        &x_data,
        &y_data,
        1,
        None,
        ProviderPolyfitOutputs::Statistics,
    )
    .expect_err("requested original-basis R cannot be represented");
    assert!(matches!(error, PolyfitKernelError::InvalidInput(_)));

    let provider = Box::leak(Box::new(SolverProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        calls: AtomicUsize::new(0),
        override_result: Mutex::new(None),
    }));
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let x = provider
        .inner
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &x_data,
            shape: &[2, 1],
        })
        .unwrap();
    let y = provider
        .inner
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &y_data,
            shape: &[2, 1],
        })
        .unwrap();
    let args = [
        Value::GpuTensor(x.clone()),
        Value::GpuTensor(y.clone()),
        Value::Num(1.0),
    ];
    let result = futures::executor::block_on(runmat_runtime::call_builtin_async("polyfit", &args))
        .expect("one-output provider fit");
    let Value::Tensor(coefficients) = result.value() else {
        panic!("expected real coefficient tensor");
    };
    assert!((coefficients.materialize_f64()[0] * f64::MAX - 1.0).abs() < 1e-14);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    x.allocation_lease().validate_live().unwrap();
    y.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn coefficient_only_fit_does_not_form_unrepresentable_residual_norm() {
    let x = [0.0, 1.0];
    let y = [f64::MAX, -f64::MAX];
    let result =
        polyfit_host_real_for_provider(&x, &y, 0, None, ProviderPolyfitOutputs::Coefficients)
            .expect("finite constant coefficient");
    assert_eq!(result.coefficients.len(), 1);
    assert!(result.coefficients[0].is_finite());
    assert!(result.coefficients[0].abs() <= f64::MAX * 1e-14);
    assert!(result.r_matrix.is_none());
    assert!(result.normr.is_none());
    let error = polyfit_host_real_for_provider(&x, &y, 0, None, ProviderPolyfitOutputs::Statistics)
        .expect_err("requested residual exceeds binary64 range");
    assert!(matches!(error, PolyfitKernelError::InvalidInput(_)));
    assert!(error.to_string().contains("residual norm"));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn malformed_provider_success_is_operational_and_preserves_borrowed_inputs() {
    let provider = Box::leak(Box::new(SolverProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        calls: AtomicUsize::new(0),
        override_result: Mutex::new(None),
    }));
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let x = provider
        .inner
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &[0.0, 1.0],
            shape: &[2, 1],
        })
        .unwrap();
    let y = provider
        .inner
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &[1.0, 3.0],
            shape: &[2, 1],
        })
        .unwrap();
    let args = [
        Value::GpuTensor(x.clone()),
        Value::GpuTensor(y.clone()),
        Value::Num(1.0),
    ];
    let base = runmat_accelerate_api::ProviderPolyfitResult {
        coefficients: vec![2.0, 1.0],
        r_matrix: None,
        normr: None,
        df: 0.0,
        mu: [0.0, 1.0],
    };
    for malformed in [
        runmat_accelerate_api::ProviderPolyfitResult {
            coefficients: vec![f64::NAN, 1.0],
            ..base.clone()
        },
        runmat_accelerate_api::ProviderPolyfitResult {
            r_matrix: Some(vec![1.0; 4]),
            ..base.clone()
        },
        runmat_accelerate_api::ProviderPolyfitResult {
            normr: Some(f64::NAN),
            ..base.clone()
        },
        runmat_accelerate_api::ProviderPolyfitResult {
            df: 1.0,
            ..base.clone()
        },
        runmat_accelerate_api::ProviderPolyfitResult {
            mu: [f64::NAN, 1.0],
            ..base.clone()
        },
    ] {
        *provider.override_result.lock().unwrap() = Some(malformed);
        let error =
            futures::executor::block_on(runmat_runtime::call_builtin_async("polyfit", &args))
                .expect_err("malformed provider success must not publish or gather");
        assert_eq!(error.identifier(), Some("RunMat:polyfit:ProviderFailed"));
        assert_eq!(
            error.gpu_gather_retry(),
            runmat_runtime::GpuGatherRetry::Never
        );
        assert!(
            error.source().is_some(),
            "typed validation cause must be retained"
        );
        x.allocation_lease().validate_live().unwrap();
        y.allocation_lease().validate_live().unwrap();
    }
    assert_eq!(provider.calls.load(Ordering::SeqCst), 5);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn public_weight_errors_match_host_and_retain_provider_cause_and_inputs() {
    let provider = Box::leak(Box::new(SolverProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        calls: AtomicUsize::new(0),
        override_result: Mutex::new(None),
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
