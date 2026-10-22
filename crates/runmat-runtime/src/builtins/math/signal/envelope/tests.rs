use super::*;
use futures::executor::block_on;
use runmat_value::IntegerStorage;

fn row(values: Vec<f64>) -> Value {
    Value::Tensor(Tensor::new(values.clone(), vec![1, values.len()]).expect("tensor"))
}

fn col(values: Vec<f64>) -> Value {
    Value::Tensor(Tensor::new(values.clone(), vec![values.len(), 1]).expect("tensor"))
}

fn integer_row(values: Vec<i16>) -> Value {
    let tensor = Tensor::new_integer(IntegerStorage::I16(values.clone()), vec![1, values.len()])
        .expect("typed integer row");
    Value::Tensor(tensor)
}

fn integer_matrix(values: Vec<i16>, shape: Vec<usize>) -> Value {
    let tensor =
        Tensor::new_integer(IntegerStorage::I16(values), shape).expect("typed integer matrix");
    Value::Tensor(tensor)
}

fn call(value: Value, rest: Vec<Value>, outputs: Option<usize>) -> BuiltinResult<Value> {
    let _guard = outputs
        .map(|count| crate::output_count::push_output_count(Some(count)))
        .transpose()
        .expect("output-count scope");
    block_on(envelope_builtin(
        None,
        value,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
}

fn tensor_data(value: Value) -> Vec<f64> {
    match value {
        Value::Tensor(tensor) => tensor.materialize_f64(),
        Value::Num(value) => vec![value],
        other => panic!("expected numeric output, got {other:?}"),
    }
}

fn output_list(value: Value) -> Vec<Value> {
    let Value::OutputList(values) = value else {
        panic!("expected output list, got {value:?}");
    };
    values
        .try_into_owned()
        .expect("unique output list")
        .try_into_unbounded_vec()
        .expect("unbounded output list")
}

#[test]
fn analytic_envelope_tracks_am_modulation() {
    let n = 128usize;
    let values = (0..n)
        .map(|idx| {
            let t = idx as f64 / n as f64;
            let carrier = (2.0 * std::f64::consts::PI * 8.0 * t).sin();
            let amp = 1.25 + 0.5 * (2.0 * std::f64::consts::PI * t).sin();
            amp * carrier
        })
        .collect::<Vec<_>>();

    let out = call(row(values), Vec::new(), None).expect("envelope");
    let upper = tensor_data(out);
    for (idx, actual) in upper.iter().enumerate().skip(4).take(n - 8) {
        let t = idx as f64 / n as f64;
        let expected = 1.25 + 0.5 * (2.0 * std::f64::consts::PI * t).sin();
        assert!(
            (actual - expected).abs() < 0.15,
            "idx={idx} actual={actual} expected={expected}"
        );
    }
}

#[test]
fn analytic_envelope_adds_mean_back_to_bounds() {
    let values = vec![3.0, 4.0, 3.0, 2.0];
    let outputs = output_list(call(row(values), Vec::new(), Some(2)).expect("envelope"));
    let upper = tensor_data(outputs[0].clone());
    let lower = tensor_data(outputs[1].clone());
    assert_eq!(upper.len(), 4);
    assert_eq!(lower.len(), 4);
    assert!(upper.iter().all(|value| *value >= 3.0));
    assert!(lower.iter().all(|value| *value <= 3.0));
}

#[test]
fn envelope_reads_typed_integer_vector_storage_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let outputs = output_list(call(integer_row(vec![3, 4, 3, 2]), Vec::new(), Some(2)).unwrap());
    let upper = tensor_data(outputs[0].clone());
    let lower = tensor_data(outputs[1].clone());
    assert_eq!(upper.len(), 4);
    assert_eq!(lower.len(), 4);
    assert!(upper.iter().all(|value| value.is_finite()));
    assert!(lower.iter().all(|value| value.is_finite()));
    assert!(upper.iter().all(|value| *value >= 3.0));
    assert!(lower.iter().all(|value| *value <= 3.0));
}

#[test]
fn envelope_preserves_native_single_output_on_host() {
    let input =
        Tensor::from_numeric_storage(NumericStorage::F32(vec![3.0, 4.0, 3.0, 2.0]), vec![1, 4])
            .expect("single input");
    let outputs =
        output_list(call(Value::Tensor(input), Vec::new(), Some(2)).expect("single envelope"));
    for output in outputs {
        let Value::Tensor(tensor) = output else {
            panic!("expected single tensor output");
        };
        assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
    }
}

#[test]
fn envelope_integer_data_and_control_are_separately_gated() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let data_error = call(integer_row(vec![1, 2, 3]), Vec::new(), Some(1))
        .expect_err("typed integer data is an extension");
    assert_eq!(
        data_error.identifier(),
        ENVELOPE_INTEGER_DATA_EXTENSION.error_identifier
    );

    let control_error = call(
        row(vec![1.0, 2.0, 3.0]),
        vec![Value::Int(runmat_value::IntValue::U64(3))],
        Some(1),
    )
    .expect_err("typed integer control is an extension");
    assert_eq!(
        control_error.identifier(),
        ENVELOPE_INTEGER_CONTROL_EXTENSION.error_identifier
    );
}

#[test]
fn envelope_rejects_lossy_wide_integer_before_floating_arithmetic() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let error = call(
        Value::Int(runmat_value::IntValue::U64(9_007_199_254_740_993)),
        Vec::new(),
        Some(1),
    )
    .expect_err("inexact binary64 signal boundary");
    assert!(error.message().contains("exactly representable"));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
}

#[test]
fn host_envelope_rejects_nan_and_infinities_with_the_signal_descriptor() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let error = call(col(vec![1.0, invalid, 3.0]), Vec::new(), Some(1))
            .expect_err("nonfinite host signal");
        assert_eq!(error.identifier(), ENVELOPE_ERROR_INVALID_SIGNAL.identifier);
    }
}

#[test]
fn typed_resident_nonfinite_cause_is_not_an_unavailable_hook() {
    let error = runmat_accelerate_api::ProviderExecutionError::from_source(
        runmat_accelerate_api::EnvelopeSignalDomainError::NonFinite,
    )
    .after_effect();
    assert!(!error.is_unsupported());
    assert!(error.is_after_effect());
    assert!(provider_rejected_nonfinite_signal(&error));
}

#[test]
fn host_envelope_abandons_unpublished_bounds_on_cancellation() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::num::NonZeroU64;

    let source = CancellationSource::new();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(4).unwrap()),
        source.token(),
        move || {
            source.cancel(CancellationReason::User);
            Box::pin(async { Ok(()) })
        },
    );
    let mut cooperation = SignalCooperation::with_budget(budget, BUILTIN_NAME);
    let input = SignalInput {
        data: vec![1.0, 2.0, 3.0, 4.0],
        shape: vec![4, 1],
        rows: 4,
        cols: 1,
        vector: true,
        single_output: false,
    };
    let error = block_on(evaluate(
        input,
        &EnvelopeOptions {
            method: EnvelopeMethod::Rms,
            parameter: Some(2),
        },
        &mut cooperation,
    ))
    .expect_err("completed work must not publish after cancellation");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
}

#[test]
fn one_large_channel_checks_each_numerical_method_before_publication() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::cell::Cell;
    use std::num::NonZeroU64;
    use std::rc::Rc;

    for method in [
        EnvelopeMethod::Analytic,
        EnvelopeMethod::Rms,
        EnvelopeMethod::Peak,
    ] {
        let source = CancellationSource::new();
        let turns = Rc::new(Cell::new(0));
        let observed = turns.clone();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(
                NonZeroU64::new(super::super::common::SIGNAL_WORK_CHUNK).unwrap(),
            ),
            source.token(),
            move || {
                observed.set(observed.get() + 1);
                source.cancel(CancellationReason::User);
                Box::pin(async { Ok(()) })
            },
        );
        let mut cooperation = SignalCooperation::with_budget(budget, BUILTIN_NAME);
        let input = SignalInput {
            data: vec![1.0; super::super::common::SIGNAL_WORK_CHUNK as usize + 1],
            shape: vec![super::super::common::SIGNAL_WORK_CHUNK as usize + 1, 1],
            rows: super::super::common::SIGNAL_WORK_CHUNK as usize + 1,
            cols: 1,
            vector: true,
            single_output: false,
        };
        let error = block_on(evaluate(
            input,
            &EnvelopeOptions {
                method,
                parameter: Some(3),
            },
            &mut cooperation,
        ))
        .expect_err("one large channel must observe cancellation during numerical work");
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert_eq!(turns.get(), 1);
    }
}

#[test]
fn envelope_resident_integer_data_is_gated_before_provider_access() {
    use crate::builtins::common::{gpu_helpers, test_support};

    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
            vec![1, 2],
        )
        .expect("integer tensor");
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
        provider.reset_telemetry();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = call(Value::GpuTensor(handle.clone()), Vec::new(), Some(1))
            .expect_err("resident integer data is an extension");
        assert_eq!(
            error.identifier(),
            ENVELOPE_INTEGER_DATA_EXTENSION.error_identifier
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        handle.release().expect("free input");
    });
}

#[test]
fn rms_envelope_uses_centered_window() {
    let outputs = output_list(
        call(
            row(vec![0.0, 3.0, 4.0, 0.0]),
            vec![Value::Num(3.0), Value::String("rms".to_string().into())],
            Some(2),
        )
        .expect("envelope"),
    );
    let upper = tensor_data(outputs[0].clone());
    let lower = tensor_data(outputs[1].clone());
    assert!((upper[1] - (25.0f64 / 3.0).sqrt()).abs() < 1.0e-12);
    assert_eq!(lower[1], -upper[1]);
}

#[test]
fn peak_envelope_interpolates_extrema() {
    let outputs = output_list(
        call(
            row(vec![0.0, 2.0, 0.0, -1.0, 0.0, 3.0, 0.0]),
            vec![Value::Num(2.0), Value::String("peak".to_string().into())],
            Some(2),
        )
        .expect("envelope"),
    );
    let upper = tensor_data(outputs[0].clone());
    let lower = tensor_data(outputs[1].clone());
    assert!((upper[1] - 2.0).abs() < 1.0e-12);
    assert!((upper[5] - 3.0).abs() < 1.0e-12);
    assert!((lower[3] + 1.0).abs() < 1.0e-12);
}

#[test]
fn matrix_inputs_are_processed_column_wise() {
    let matrix =
        Tensor::new(vec![1.0, 0.0, -1.0, 0.0, 2.0, 3.0, 2.0, 1.0], vec![4, 2]).expect("matrix");
    let out = call(Value::Tensor(matrix), Vec::new(), Some(2)).expect("envelope");
    let outputs = output_list(out);
    let Value::Tensor(upper) = outputs[0].clone() else {
        panic!("expected tensor");
    };
    assert_eq!(upper.shape(), [4, 2]);
    assert!(upper.materialize_f64()[..4]
        .iter()
        .all(|value| (*value - 1.0).abs() < 1.0e-12));
    assert!(upper.materialize_f64()[4..]
        .iter()
        .all(|value| *value >= 2.0));
}

#[test]
fn envelope_reads_typed_integer_matrix_storage_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = call(
        integer_matrix(vec![1, 0, -1, 0, 2, 3, 2, 1], vec![4, 2]),
        Vec::new(),
        Some(2),
    )
    .expect("envelope");
    let outputs = output_list(out);
    let Value::Tensor(upper) = outputs[0].clone() else {
        panic!("expected tensor");
    };
    assert_eq!(upper.shape(), [4, 2]);
    assert!(upper.materialize_f64()[..4]
        .iter()
        .all(|value| (*value - 1.0).abs() < 1.0e-12));
    assert!(upper.materialize_f64()[4..]
        .iter()
        .all(|value| *value >= 2.0));
}

#[cfg(feature = "plot-core")]
#[test]
fn output_count_zero_returns_empty_output_list() {
    crate::builtins::plotting::tests::ensure_plot_test_env();
    let _plot_lock = crate::builtins::plotting::tests::lock_plot_registry();
    let out = call(col(vec![1.0, 0.0, -1.0, 0.0]), Vec::new(), Some(0)).expect("envelope");
    assert_eq!(out, Value::output_list(Vec::new()));
}

#[cfg(not(feature = "plot-core"))]
#[test]
fn output_count_zero_preserves_host_validation_before_typed_missing_presentation() {
    let invalid = call(Value::from("not a signal"), Vec::new(), Some(0))
        .expect_err("known invalid host input");
    assert_eq!(
        invalid.identifier(),
        ENVELOPE_ERROR_INVALID_SIGNAL.identifier
    );
    let unavailable = call(col(vec![1.0, 0.0, -1.0, 0.0]), Vec::new(), Some(0))
        .expect_err("presentation needs plot-core");
    assert!(unavailable
        .source
        .as_deref()
        .and_then(|source| source.downcast_ref::<crate::plotting_hooks::PlottingUnavailable>())
        .is_some());
}

#[cfg(not(feature = "plot-core"))]
#[test]
fn resident_statement_preflight_rejects_known_shape_and_storage_without_download() {
    use runmat_accelerate_api::{
        test_support::TestAllocationOwner, AccelDownloadFuture, AccelProvider, GpuTensorHandle,
        GpuTensorStorage, HostTensorView, NumericElementType, ThreadProviderGuard,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountedProvider {
        owner: TestAllocationOwner,
        downloads: AtomicUsize,
    }
    impl AccelProvider for CountedProvider {
        fn device_id(&self) -> u32 {
            self.owner.device_id()
        }
        fn device_info(&self) -> String {
            "envelope preflight fixture".into()
        }
        fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
            panic!("statement preflight must not upload")
        }
        fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            self.downloads.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { panic!("statement preflight must not download") })
        }
    }

    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(CountedProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _active = ThreadProviderGuard::set(Some(provider));
    for (shape, storage) in [
        (vec![0, 1], GpuTensorStorage::Real),
        (vec![2, 2, 2], GpuTensorStorage::Real),
        (vec![2, 1], GpuTensorStorage::ComplexInterleaved),
    ] {
        let allocation = provider
            .owner
            .allocate_zeroed(shape, NumericElementType::F64, storage)
            .expect("resident fixture");
        for output_count in [0, 1] {
            let error = call(
                Value::GpuTensor(allocation.handle.clone()),
                Vec::new(),
                Some(output_count),
            )
            .expect_err("known invalid resident metadata");
            assert_eq!(error.identifier(), ENVELOPE_ERROR_INVALID_SIGNAL.identifier);
        }
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        allocation
            .handle
            .allocation_lease()
            .validate_live()
            .unwrap();
        assert_eq!(allocation.probe.retirements(), 0);
    }
}

#[cfg(feature = "plot-core")]
#[test]
fn matrix_statement_form_plots_each_channel_without_length_mismatch() {
    crate::builtins::plotting::tests::ensure_plot_test_env();
    let _plot_lock = crate::builtins::plotting::tests::lock_plot_registry();
    let matrix =
        Tensor::new(vec![1.0, 0.0, -1.0, 0.0, 2.0, 3.0, 2.0, 1.0], vec![4, 2]).expect("matrix");
    let out = call(Value::Tensor(matrix), Vec::new(), Some(0)).expect("envelope");
    assert_eq!(out, Value::output_list(Vec::new()));
}

#[test]
fn analytic_filter_length_uses_fir_path() {
    let values = vec![0.0, 1.0, 0.0, -0.5, 0.0, 0.25, 0.0, -0.125];
    let dft = tensor_data(call(row(values.clone()), Vec::new(), None).expect("envelope"));
    let fir = tensor_data(
        call(
            row(values),
            vec![
                Value::Num(5.0),
                Value::String("analytic".to_string().into()),
            ],
            None,
        )
        .expect("envelope"),
    );
    assert_eq!(dft.len(), fir.len());
    assert!(
        dft.iter()
            .zip(fir.iter())
            .any(|(left, right)| (left - right).abs() > 1.0e-6),
        "filter-length analytic form should not silently use the DFT path"
    );
}

#[cfg(feature = "wgpu")]
fn assert_gpu_envelope_matches_cpu(values: Vec<f64>, shape: Vec<usize>, rest: Vec<Value>) {
    use crate::builtins::common::test_support;
    use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
    use runmat_accelerate_api::HostTensorView;

    let _guard = test_support::accel_test_lock();
    if register_wgpu_provider(WgpuProviderOptions::default()).is_err() {
        return;
    }
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");
    let tensor = Tensor::new(values.clone(), shape.clone()).expect("cpu tensor");
    let expected =
        output_list(call(Value::Tensor(tensor), rest.clone(), Some(2)).expect("cpu envelope"));
    let handle = provider
        .upload(&HostTensorView {
            data: &values,
            shape: &shape,
        })
        .expect("upload");
    let actual =
        output_list(call(Value::GpuTensor(handle.clone()), rest, Some(2)).expect("gpu envelope"));
    let (Value::GpuTensor(upper_handle), Value::GpuTensor(lower_handle)) =
        (actual[0].clone(), actual[1].clone())
    else {
        panic!("expected resident gpu outputs, got {actual:?}");
    };
    let upper = test_support::gather(Value::GpuTensor(upper_handle.clone())).expect("upper");
    let lower = test_support::gather(Value::GpuTensor(lower_handle.clone())).expect("lower");
    let expected_upper = tensor_data(expected[0].clone());
    let expected_lower = tensor_data(expected[1].clone());
    assert_eq!(upper.shape(), shape);
    assert_eq!(lower.shape(), shape);
    for (actual, expected) in upper.materialize_f64().iter().zip(expected_upper.iter()) {
        assert!((actual - expected).abs() < 1.0e-5, "{actual} != {expected}");
    }
    for (actual, expected) in lower.materialize_f64().iter().zip(expected_lower.iter()) {
        assert!((actual - expected).abs() < 1.0e-5, "{actual} != {expected}");
    }
    handle.release().ok();
    upper_handle.release().ok();
    lower_handle.release().ok();
    runmat_accelerate::simple_provider::register_inprocess_provider();
}

#[cfg(feature = "wgpu")]
#[test]
fn analytic_envelope_wgpu_stays_resident_and_matches_cpu() {
    let values = (0..16)
        .map(|idx| (idx as f64 * 0.7).sin() + 0.25)
        .collect::<Vec<_>>();
    assert_gpu_envelope_matches_cpu(values, vec![1, 16], Vec::new());
}

#[cfg(feature = "wgpu")]
#[test]
fn analytic_fir_envelope_wgpu_stays_resident_and_matches_cpu() {
    let values = vec![0.0, 1.0, 0.0, -0.5, 0.0, 0.25, 0.0, -0.125];
    assert_gpu_envelope_matches_cpu(
        values,
        vec![8, 1],
        vec![
            Value::Num(5.0),
            Value::String("analytic".to_string().into()),
        ],
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn rms_envelope_wgpu_stays_resident_and_matches_cpu() {
    let values = vec![0.0, 3.0, 4.0, 0.0, 1.0, -2.0, 0.5, 0.25];
    assert_gpu_envelope_matches_cpu(
        values,
        vec![4, 2],
        vec![Value::Num(3.0), Value::String("rms".to_string().into())],
    );
}

#[test]
fn too_many_outputs_has_stable_identifier() {
    let err = call(row(vec![1.0, 0.0, -1.0, 0.0]), Vec::new(), Some(3))
        .expect_err("too many outputs should fail");
    assert_eq!(err.identifier(), ENVELOPE_ERROR_TOO_MANY_OUTPUTS.identifier);
}

#[test]
fn invalid_method_has_stable_identifier() {
    let err = call(
        row(vec![1.0, 2.0, 3.0]),
        vec![Value::Num(3.0), Value::String("bad".to_string().into())],
        None,
    )
    .expect_err("invalid method should fail");
    assert_eq!(err.identifier(), ENVELOPE_ERROR_INVALID_METHOD.identifier);
}
