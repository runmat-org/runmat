//! Public filtfilt contract and provider route witnesses.

use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::{
    handle_precision, provider_for_handle, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, AccelProviderFuture, GpuTensorHandle, HostTensorView, ProviderIirFilterOptions,
    ProviderIirFilterResult, ProviderPrecision,
};
use runmat_builtins::builtin_function_by_name;
use runmat_value::{ComplexTensor, Tensor};

fn call(b: Value, a: Value, x: Value) -> BuiltinResult<Value> {
    block_on(evaluate(b, a, x))
}

fn tensor_data(value: Value) -> (Vec<f64>, Vec<usize>) {
    let tensor = test_support::gather(value).expect("gather tensor");
    (tensor.materialize_f64(), tensor.shape)
}

fn approx(lhs: &[f64], rhs: &[f64], tol: f64) {
    assert_eq!(lhs.len(), rhs.len());
    for (idx, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
        assert!((a - b).abs() <= tol, "mismatch at {idx}: got {a}, want {b}");
    }
}

#[test]
fn prepared_host_staged_row_pass_uses_order_first_multistate_geometry() {
    let _lock = test_support::accel_test_lock();
    let provider: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
        Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
    );
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let coeffs = NormalizedCoefficients::new(
        &[
            Complex::new(0.25, 0.0),
            Complex::new(0.5, 0.0),
            Complex::new(0.25, 0.0),
        ],
        &[Complex::new(1.0, 0.0)],
    )
    .unwrap();
    let prepared = ProviderPreparedIirCoefficients::upload(
        provider,
        &[0.25, 0.5, 0.25],
        &[1.0, 0.0, 0.0],
        provider.precision(),
        &[],
    )
    .unwrap();
    let signal = [
        Complex::new(1.0, 0.0),
        Complex::new(2.0, 0.0),
        Complex::new(3.0, 0.0),
        Complex::new(4.0, 0.0),
    ];
    let output = block_on(gpu_filter_pass(
        provider,
        &prepared,
        &coeffs,
        &signal,
        &[1, 4],
        &[Complex::new(0.0, 0.0); 2],
        &[],
    ))
    .expect("prepared row pass must admit order-first state");
    let actual = output.into_iter().map(|value| value.re).collect::<Vec<_>>();
    approx(&actual, &[0.25, 1.0, 2.0, 3.0], 1e-12);
}

fn value_tolerance(value: &Value) -> f64 {
    match value {
        Value::GpuTensor(handle) => match handle_precision(handle)
            .or_else(|| provider_for_handle(handle).map(|provider| provider.precision()))
            .unwrap_or(ProviderPrecision::F64)
        {
            ProviderPrecision::F64 => 1e-8,
            ProviderPrecision::F32 => 1e-6,
        },
        _ => 1e-8,
    }
}

struct DownloadFailProvider {
    owner: runmat_accelerate_api::test_support::TestAllocationOwner,
    probes: std::sync::Mutex<Vec<runmat_accelerate_api::test_support::RetirementProbe>>,
}
impl DownloadFailProvider {
    fn new() -> Self {
        Self {
            owner: Default::default(),
            probes: Default::default(),
        }
    }
    fn free_count(&self) -> usize {
        self.probes
            .lock()
            .unwrap()
            .iter()
            .map(|probe| probe.retirements())
            .sum()
    }
}
impl AccelProvider for DownloadFailProvider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        let allocation =
            self.owner
                .allocate_numeric(runmat_accelerate_api::HostNumericTensorOwned {
                    data: runmat_accelerate_api::HostNumericDataOwned::F64(host.data.to_vec()),
                    shape: host.shape.to_vec(),
                    storage: runmat_accelerate_api::GpuTensorStorage::Real,
                })?;
        self.probes.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }
    fn download<'a>(&'a self, _h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { Err(anyhow::anyhow!("forced download failure")) })
    }
    fn device_info(&self) -> String {
        "download-fail-provider".to_string()
    }
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
}

#[test]
fn descriptor_is_registered() {
    assert!(builtin_function_by_name(BUILTIN_NAME).is_none());
    let builtin = runmat_builtins::builtin_catalog_entry_by_name(BUILTIN_NAME)
        .expect("canonical filtfilt builtin");
    let descriptor = builtin.descriptor;
    assert_eq!(builtin.bindings, &runmat_builtins::REQUIRED_DEFAULT_BINDING);
    let bindings = crate::builtin::runtime_builtin_bindings_by_name(BUILTIN_NAME);
    assert_eq!(bindings.len(), 1);
    assert_eq!(
        bindings[0].identity,
        builtin.binding_identity(&builtin.bindings[0])
    );
    assert_eq!(descriptor.signatures[0].label, "y = filtfilt(b, a, x)");
    assert!(descriptor
        .errors
        .iter()
        .any(|err| err.code == "RM.FILTFILT.INVALID_SIGNAL"));
    assert_eq!(builtin.integer_capabilities.len(), 1);
    assert_eq!(builtin.extensions.len(), 2);
}

#[test]
fn registered_filtfilt_binding_uses_invocation_cancellation() {
    let source = runmat_execution::cancellation::CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    );
    source.cancel(runmat_execution::CancellationReason::User);
    let error = block_on(filtfilt_builtin(
        Some(runtime),
        Value::Num(1.0),
        Value::Num(1.0),
        Value::Num(1.0),
    ))
    .expect_err("cancelled invocation");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
}

#[test]
fn operational_source_keeps_filtfilt_as_the_owning_builtin() {
    let mut source = build_runtime_error("readback failed")
        .with_builtin("gather")
        .with_identifier("RunMat:gather:ProviderFailed")
        .build();
    source.gpu_gather_retry = crate::GpuGatherRetry::Never;
    let error = filtfilt_error_with_source(
        &FILTFILT_ERROR_PASS_FAILED,
        "provider result gather failed",
        source,
        true,
    );
    assert_eq!(error.context.builtin.as_deref(), Some(BUILTIN_NAME));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    let nested = std::error::Error::source(&error).expect("original gather cause");
    assert!(nested.to_string().contains("readback failed"));
}

#[test]
fn final_provider_output_rejects_borrowed_alias_and_wrong_shape() {
    let _lock = test_support::accel_test_lock();
    let provider: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
        Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
    );
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let signal = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0],
            shape: &[1, 2],
        })
        .unwrap();
    let borrowed = provider
        .upload(&HostTensorView {
            data: &[3.0, 4.0],
            shape: &[1, 2],
        })
        .unwrap();
    let protected = [&borrowed];
    let context = OutputContext {
        input: &signal,
        provider,
        caller_inputs: &protected,
    };
    let contract = resident_output::floating_contract(
        &signal,
        vec![1, 2],
        runmat_accelerate_api::GpuTensorStorage::Real,
    );
    let alias_error = context
        .admit(borrowed.clone(), &contract, &[], || {
            filtfilt_error(&FILTFILT_ERROR_PROVIDER_OUTPUT)
        })
        .expect_err("a borrowed coefficient may not be published as the result");
    assert_eq!(
        alias_error.identifier(),
        FILTFILT_ERROR_PROVIDER_OUTPUT.identifier
    );
    assert!(borrowed.allocation_lease().validate_live().is_ok());
    let wrong_shape = provider
        .upload(&HostTensorView {
            data: &[5.0, 6.0],
            shape: &[2, 1],
        })
        .unwrap();
    let shape_error = context
        .admit(wrong_shape, &contract, &[], || {
            filtfilt_error(&FILTFILT_ERROR_PROVIDER_OUTPUT)
        })
        .expect_err("same-count transposed result must be rejected");
    assert_eq!(
        shape_error.identifier(),
        FILTFILT_ERROR_PROVIDER_OUTPUT.identifier
    );
    assert!(signal.allocation_lease().validate_live().is_ok());
}

#[test]
fn compatibility_mode_rejects_integer_and_logical_extensions() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let integer = call(
        Value::Int(runmat_value::IntValue::I16(1)),
        Value::Num(1.0),
        Value::Num(1.0),
    )
    .expect_err("integer extension must reject");
    assert_eq!(
        integer.identifier(),
        FILTFILT_INTEGER_INPUT_EXTENSION.error_identifier
    );
    let logical = call(Value::Bool(true), Value::Num(1.0), Value::Num(1.0))
        .expect_err("logical extension must reject");
    assert_eq!(
        logical.identifier(),
        FILTFILT_LOGICAL_INPUT_EXTENSION.error_identifier
    );
}

#[test]
fn nonzero_subnormal_denominator_is_not_classified_as_zero() {
    let signal = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let output = call(
        Value::Num(1.0e-300),
        Value::Num(1.0e-300),
        Value::Tensor(signal),
    )
    .expect("finite nonzero denominator");
    let (values, shape) = tensor_data(output);
    assert_eq!(shape, vec![1, 2]);
    assert_eq!(values, vec![1.0, 2.0]);
}

#[test]
fn tiny_nonzero_imaginary_coefficient_remains_complex() {
    let signal = Tensor::new((1..=8).map(f64::from).collect(), vec![1, 8]).unwrap();
    let numerator = ComplexTensor::new(vec![(1.0, 0.0), (0.0, 1.0e-300)], vec![1, 2]).unwrap();
    let output = call(
        Value::ComplexTensor(numerator),
        Value::Num(1.0),
        Value::Tensor(signal),
    )
    .expect("small imaginary coefficient");
    let Value::ComplexTensor(output) = output else {
        panic!("complex coefficient must produce complex output");
    };
    assert_eq!(output.shape, vec![1, 8]);
    assert!(output.materialize_f64().iter().any(|&(_, im)| im != 0.0));
}

#[test]
fn finite_large_constant_reflection_does_not_overflow_before_identity_filter() {
    let signal = Tensor::new(vec![1.0e308; 8], vec![1, 8]).unwrap();
    let identity_with_state = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
    let output = call(
        Value::Tensor(identity_with_state),
        Value::Num(1.0),
        Value::Tensor(signal),
    )
    .expect("representable large constant reflection");
    let (values, shape) = tensor_data(output);
    assert_eq!(shape, vec![1, 8]);
    assert_eq!(values, vec![1.0e308; 8]);

    let signal = Tensor::new(
        vec![f64::MAX, -f64::MAX, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![1, 8],
    )
    .unwrap();
    let b = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
    let error = call(Value::Tensor(b), Value::Num(1.0), Value::Tensor(signal))
        .expect_err("unrepresentable endpoint reflection");
    assert_eq!(error.identifier(), FILTFILT_ERROR_INVALID_SIGNAL.identifier);
}

#[test]
fn resident_exact_wide_integer_uses_authoritative_gather_fallback() {
    test_support::with_test_provider(|provider| {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let signal = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_994; 4]),
            vec![1, 4],
        )
        .expect("wide signal");
        let handle = gpu_helpers::upload_tensor(provider, &signal).expect("wide upload");
        let result = block_on(evaluate(
            Value::Num(1.0),
            Value::Num(1.0),
            Value::GpuTensor(handle),
        ))
        .expect("exact resident wide integer");
        let gathered = test_support::gather(result).expect("gather result");
        assert_eq!(gathered.numeric_dtype(), NumericDType::F64);
        assert_eq!(gathered.shape, vec![1, 4]);
    });
}

#[test]
fn single_output_is_native_and_nonfinite_signal_rejects() {
    let single = Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let value = call(Value::Num(1.0), Value::Num(1.0), Value::Tensor(single)).unwrap();
    let Value::Tensor(value) = value else {
        panic!("expected tensor")
    };
    assert_eq!(value.numeric_dtype(), NumericDType::F32);

    let nonfinite = Tensor::new(vec![1.0, f64::NAN], vec![1, 2]).unwrap();
    call(Value::Num(1.0), Value::Num(1.0), Value::Tensor(nonfinite))
        .expect_err("nonfinite signal must reject");
}

#[test]
fn fir_moving_average_preserves_row_shape() {
    let b = Tensor::new(vec![1.0 / 3.0; 3], vec![1, 3]).unwrap();
    let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let x = Tensor::new((1..=20).map(|v| v as f64).collect(), vec![1, 20]).unwrap();
    let (data, shape) =
        tensor_data(call(Value::Tensor(b), Value::Tensor(a), Value::Tensor(x)).unwrap());
    assert_eq!(shape, vec![1, 20]);
    approx(
        &data[3..17],
        &(4..=17).map(|v| v as f64).collect::<Vec<_>>(),
        1e-9,
    );
}

#[test]
fn iir_coefficients_are_supported() {
    let b = Tensor::new(vec![0.2], vec![1, 1]).unwrap();
    let a = Tensor::new(vec![1.0, -0.8], vec![1, 2]).unwrap();
    let x = Tensor::new(vec![1.0; 16], vec![16, 1]).unwrap();
    let (data, shape) =
        tensor_data(call(Value::Tensor(b), Value::Tensor(a), Value::Tensor(x)).unwrap());
    assert_eq!(shape, vec![16, 1]);
    approx(&data, &[1.0; 16], 1e-8);
}

#[test]
fn rejects_too_short_signal() {
    let b = Tensor::new(vec![1.0 / 3.0; 3], vec![1, 3]).unwrap();
    let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
    let err = call(Value::Tensor(b), Value::Tensor(a), Value::Tensor(x)).unwrap_err();
    assert!(err.message().contains("greater than"));
}

#[test]
fn scalar_gain_filters_forward_and_backward() {
    let y = call(Value::Num(2.0), Value::Num(1.0), Value::Num(3.0)).unwrap();
    let (data, shape) = tensor_data(y);
    assert_eq!(shape, vec![1, 1]);
    approx(&data, &[12.0], 1e-12);
}

#[test]
fn complex_host_path() {
    let b = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let x = ComplexTensor::new(vec![(1.0, 1.0), (2.0, -1.0)], vec![1, 2]).unwrap();
    let value = call(Value::Tensor(b), Value::Tensor(a), Value::ComplexTensor(x)).unwrap();
    let Value::ComplexTensor(out) = value else {
        panic!("expected complex output");
    };
    assert_eq!(out.shape, vec![1, 2]);
    assert_eq!(out.materialize_f64(), vec![(1.0, 1.0), (2.0, -1.0)]);
}

#[test]
fn gpu_filter_pass_retires_all_temporaries_when_hook_is_unsupported() {
    let _guard = test_support::accel_test_lock();
    let provider = Box::leak(Box::new(DownloadFailProvider::new()));
    let _provider_guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));

    let b = vec![Complex::new(0.5, 0.0), Complex::new(0.5, 0.0)];
    let a = vec![Complex::new(1.0, 0.0)];
    let coeffs = NormalizedCoefficients::new(&b, &a).expect("coefficients");
    let signal = (1..=8)
        .map(|value| Complex::new(value as f64, 0.0))
        .collect::<Vec<_>>();
    let zi = vec![Complex::new(0.0, 0.0)];
    let prepared = ProviderPreparedIirCoefficients::upload(
        provider,
        &[0.5, 0.5],
        &[1.0, 0.0],
        provider.precision(),
        &[],
    )
    .expect("prepared coefficients");
    let result = block_on(gpu_filter_pass(
        provider,
        &prepared,
        &coeffs,
        &signal,
        &[1, 8],
        &zi,
        &[],
    ));

    assert!(result.is_err());
    drop(prepared);
    assert_eq!(provider.free_count(), 4);
}

struct OperationalFailureProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    calls: std::sync::atomic::AtomicUsize,
    downloads: std::sync::atomic::AtomicUsize,
}

struct NoIirProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    uploads: std::sync::atomic::AtomicUsize,
}

impl AccelProvider for NoIirProvider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.uploads
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.inner.upload(host)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
        self.inner.download_numeric(handle)
    }
    fn device_info(&self) -> String {
        "filtfilt-no-iir".into()
    }
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
}

#[test]
fn absent_iir_hook_falls_back_before_coefficient_uploads() {
    let _lock = test_support::accel_test_lock();
    let provider: &'static NoIirProvider = Box::leak(Box::new(NoIirProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        uploads: std::sync::atomic::AtomicUsize::new(0),
    }));
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let x = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            shape: &[1, 8],
        })
        .expect("resident signal");
    let b = Tensor::new(vec![0.5, 0.5], vec![1, 2]).unwrap();
    let result = call(Value::Tensor(b), Value::Num(1.0), Value::GpuTensor(x))
        .expect("host fallback before unsupported provider effects");
    assert_eq!(
        provider.uploads.load(std::sync::atomic::Ordering::SeqCst),
        1
    );
    let (values, shape) = tensor_data(result);
    assert_eq!(shape, vec![1, 8]);
    approx(&values[3..7], &[4.0, 5.0, 6.0, 7.0], 1e-9);
}

#[test]
fn resident_row_filter_uses_axis_aligned_multistate_geometry() {
    let b = Tensor::new(vec![0.25, 0.5, 0.25], vec![1, 3]).unwrap();
    let values = (1..=12).map(f64::from).collect::<Vec<_>>();
    let host = call(
        Value::Tensor(b.clone()),
        Value::Num(1.0),
        Value::Tensor(Tensor::new(values.clone(), vec![1, 12]).unwrap()),
    )
    .expect("host reference");
    let (expected, expected_shape) = tensor_data(host);
    test_support::with_test_provider(|provider| {
        let signal = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &[1, 12],
            })
            .expect("resident signal");
        let result = call(Value::Tensor(b), Value::Num(1.0), Value::GpuTensor(signal))
            .expect("resident row signal with two filter states");
        assert!(matches!(result, Value::GpuTensor(_)));
        let gathered = test_support::gather(result).expect("resident result");
        assert_eq!(gathered.shape, expected_shape);
        approx(&gathered.materialize_f64(), &expected, 1e-9);
    });
}

impl AccelProvider for OperationalFailureProvider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.inner.download(handle)
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.downloads
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.inner.download_numeric(handle)
    }
    fn device_info(&self) -> String {
        "filtfilt-operational-failure".into()
    }
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn query_feasibility(
        &self,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::ProviderFeasibility {
        self.inner.query_feasibility(query)
    }
    fn iir_zero_phase<'a>(
        &'a self,
        _request: &'a runmat_accelerate_api::ProviderIirZeroPhaseRequest<'a>,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Box::pin(async {
            Err(
                runmat_accelerate_api::ProviderExecutionError::from_source(anyhow::anyhow!(
                    "device failed during zero-phase pass"
                ))
                .after_effect(),
            )
        })
    }
}

#[test]
fn resident_pass_operational_failure_keeps_cause_and_does_not_replay() {
    let _lock = test_support::accel_test_lock();
    let provider: &'static OperationalFailureProvider =
        Box::leak(Box::new(OperationalFailureProvider {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            calls: std::sync::atomic::AtomicUsize::new(0),
            downloads: std::sync::atomic::AtomicUsize::new(0),
        }));
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let x = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            shape: &[1, 8],
        })
        .unwrap();
    let b = Tensor::new(vec![0.5, 0.5], vec![1, 2]).unwrap();
    let error = call(Value::Tensor(b), Value::Num(1.0), Value::GpuTensor(x))
        .expect_err("operational pass failure must terminate");
    assert_eq!(error.identifier(), FILTFILT_ERROR_PASS_FAILED.identifier);
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(std::error::Error::source(&error).is_some());
    assert_eq!(provider.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[test]
fn host_invalid_denominator_precedes_resident_numerator_readback() {
    let _lock = test_support::accel_test_lock();
    let provider: &'static OperationalFailureProvider =
        Box::leak(Box::new(OperationalFailureProvider {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            calls: std::sync::atomic::AtomicUsize::new(0),
            downloads: std::sync::atomic::AtomicUsize::new(0),
        }));
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let b = provider
        .upload(&HostTensorView {
            data: &[0.5, 0.5],
            shape: &[1, 2],
        })
        .expect("resident coefficients");
    let empty_a = Tensor::new(Vec::new(), vec![0, 1]).expect("empty denominator");
    let error = call(
        Value::GpuTensor(b.clone()),
        Value::Tensor(empty_a),
        Value::Num(1.0),
    )
    .expect_err("known-invalid denominator");
    assert_eq!(
        error.identifier(),
        FILTFILT_ERROR_INVALID_COEFFICIENTS.identifier
    );
    assert_eq!(
        provider.downloads.load(std::sync::atomic::Ordering::SeqCst),
        0
    );
    assert_eq!(provider.calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    b.release().expect("release original resident coefficients");
}

#[test]
fn later_stale_resident_signal_precedes_numerator_download() {
    let _lock = test_support::accel_test_lock();
    let provider = crate::builtins::math::signal::common::convolution::test_support::RejectTransferProvider::new();
    let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let numerator = provider.input(vec![1.0, 1.0], vec![1, 2]);
    let signal = provider.input(vec![1.0; 8], vec![1, 8]);
    signal.probe.mark_unavailable();
    let error = call(
        Value::GpuTensor(numerator.handle.clone()),
        Value::Num(1.0),
        Value::GpuTensor(signal.handle.clone()),
    )
    .expect_err("signal lease must be checked before numerator transfer");
    assert_eq!(error.identifier(), FILTFILT_ERROR_INVALID_SIGNAL.identifier);
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(std::error::Error::source(&error).is_some());
    assert_eq!(provider.downloads(), 0);
    assert!(numerator.handle.allocation_lease().validate_live().is_ok());
    assert_eq!(numerator.probe.retirements(), 0);
}

#[test]
fn gpu_matches_cpu() {
    test_support::with_test_provider(|provider| {
        let b = Tensor::new(vec![1.0 / 3.0; 3], vec![1, 3]).unwrap();
        let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let x = Tensor::new((1..=20).map(|v| v as f64).collect(), vec![1, 20]).unwrap();
        let cpu = call(
            Value::Tensor(b.clone()),
            Value::Tensor(a.clone()),
            Value::Tensor(x.clone()),
        )
        .unwrap();
        let (cpu_data, _) = tensor_data(cpu);
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &x.materialize_f64(),
                shape: &x.shape,
            })
            .expect("upload");
        let gpu = call(Value::Tensor(b), Value::Tensor(a), Value::GpuTensor(handle)).unwrap();
        let tol = value_tolerance(&gpu);
        let (gpu_data, shape) = tensor_data(gpu);
        assert_eq!(shape, vec![1, 20]);
        approx(&gpu_data, &cpu_data, tol);
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn wgpu_matches_cpu() {
    if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_err()
    {
        return;
    }
    let Some(provider) = runmat_accelerate_api::provider() else {
        return;
    };
    let b = Tensor::new(vec![1.0 / 3.0; 3], vec![1, 3]).unwrap();
    let a = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let x = Tensor::new((1..=20).map(|v| v as f64).collect(), vec![1, 20]).unwrap();
    let cpu = call(
        Value::Tensor(b.clone()),
        Value::Tensor(a.clone()),
        Value::Tensor(x.clone()),
    )
    .unwrap();
    let (cpu_data, _) = tensor_data(cpu);
    let handle = provider
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &x.materialize_f64(),
            shape: &x.shape,
        })
        .expect("upload");
    let gpu = call(Value::Tensor(b), Value::Tensor(a), Value::GpuTensor(handle)).unwrap();
    let (gpu_data, _) = tensor_data(gpu);
    approx(&gpu_data, &cpu_data, 1e-6);
}
