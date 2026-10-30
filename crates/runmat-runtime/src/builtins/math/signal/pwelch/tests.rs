use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::AccelProvider;
use runmat_builtins::builtin_function_by_name;
use runmat_value::IntegerStorage;
use runmat_value::NumericDType;

fn call(x: Value, rest: &[Value], outputs: Option<usize>) -> BuiltinResult<Value> {
    let _guard = outputs
        .map(|count| crate::output_count::push_output_count(Some(count)))
        .transpose()
        .expect("output-count scope");
    block_on(evaluate(x, rest))
}

fn output_pair(value: Value) -> (Vec<f64>, Vec<f64>) {
    let Value::OutputList(values) = value else {
        panic!("expected output list");
    };
    let Value::Tensor(pxx) = &values[0] else {
        panic!("expected pxx tensor");
    };
    let Value::Tensor(f) = &values[1] else {
        panic!("expected f tensor");
    };
    (pxx.materialize_f64().clone(), f.materialize_f64().clone())
}

fn output_pair_with_pxx_shape(value: Value) -> (Vec<f64>, Vec<usize>, Vec<f64>) {
    let Value::OutputList(values) = value else {
        panic!("expected output list");
    };
    let Value::Tensor(pxx) = &values[0] else {
        panic!("expected pxx tensor");
    };
    let Value::Tensor(f) = &values[1] else {
        panic!("expected f tensor");
    };
    (
        pxx.materialize_f64().clone(),
        pxx.shape().to_vec(),
        f.materialize_f64().clone(),
    )
}

fn integer_tensor(values: Vec<i16>, shape: Vec<usize>) -> Tensor {
    Tensor::new_integer(IntegerStorage::I16(values), shape).expect("typed integer tensor")
}

#[test]
fn complex_controls_with_tiny_nonzero_imaginary_parts_are_not_real() {
    let x = Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![2, 1]).unwrap());
    let window = Value::ComplexTensor(
        runmat_value::ComplexTensor::new(vec![(1.0, 1e-13), (1.0, 0.0)], vec![2, 1]).unwrap(),
    );
    let error = call(
        x.clone(),
        &[window, Value::Num(0.0), Value::Num(2.0)],
        Some(1),
    )
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:pwelch:InvalidWindow"));

    let frequencies = Value::ComplexTensor(
        runmat_value::ComplexTensor::new(vec![(0.0, 0.0), (0.5, 1e-13)], vec![2, 1]).unwrap(),
    );
    let error = call(
        x,
        &[
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![2, 1]).unwrap()),
            Value::Num(0.0),
            frequencies,
        ],
        Some(1),
    )
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:pwelch:InvalidNfft"));
}

#[test]
fn wide_host_integer_exactness_uses_the_invocation_budget() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::cancellation::CancellationSource;
    use std::num::NonZeroU64;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let source = CancellationSource::new();
    let turns = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&turns);
    let mut cooperation = SignalCooperation::with_budget(
        CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            move || {
                observed.fetch_add(1, Ordering::SeqCst);
                Box::pin(async { Ok(()) })
            },
        ),
        BUILTIN_NAME,
    );
    let input = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 54; 4097]), vec![4097, 1]).unwrap(),
    );
    let receipt = block_on(ensure_pwelch_integer_boundary(
        &input,
        "signal",
        &PWELCH_ERROR_INVALID_SIGNAL,
        &PWELCH_INTEGER_SIGNAL_EXTENSION,
        &mut cooperation,
    ))
    .unwrap();
    assert!(receipt.is_none());
    assert!(turns.load(Ordering::SeqCst) >= 2);
}

#[test]
fn wide_resident_integer_signal_is_read_back_once_and_input_remains_borrowed() {
    use runmat_accelerate_api::{AccelIntegerDownloadFuture, AccelProvider, ThreadProviderGuard};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountedOwner {
        inner: runmat_accelerate::simple_provider::InProcessProvider,
        integer_downloads: AtomicUsize,
        other_downloads: AtomicUsize,
    }
    impl AccelProvider for CountedOwner {
        fn device_id(&self) -> u32 {
            self.inner.device_id()
        }
        fn device_info(&self) -> String {
            self.inner.device_info()
        }
        fn upload(
            &self,
            host: &runmat_accelerate_api::HostTensorView,
        ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
            self.inner.upload(host)
        }
        fn upload_numeric_raw(
            &self,
            host: &runmat_accelerate_api::HostNumericTensorView,
        ) -> runmat_accelerate_api::ProviderExecutionResult<runmat_accelerate_api::GpuTensorHandle>
        {
            self.inner.upload_numeric_raw(host)
        }
        fn download<'a>(
            &'a self,
            handle: &'a runmat_accelerate_api::GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            self.other_downloads.fetch_add(1, Ordering::SeqCst);
            self.inner.download(handle)
        }
        fn download_integer<'a>(
            &'a self,
            handle: &'a runmat_accelerate_api::GpuTensorHandle,
        ) -> AccelIntegerDownloadFuture<'a> {
            self.other_downloads.fetch_add(1, Ordering::SeqCst);
            self.inner.download_integer(handle)
        }
        fn download_numeric<'a>(
            &'a self,
            handle: &'a runmat_accelerate_api::GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
            self.integer_downloads.fetch_add(1, Ordering::SeqCst);
            self.inner.download_numeric(handle)
        }
    }

    let provider = Box::leak(Box::new(CountedOwner {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        integer_downloads: AtomicUsize::new(0),
        other_downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let signal =
        Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 54; 8]), vec![8, 1]).unwrap();
    let input = gpu_helpers::upload_tensor(provider, &signal).unwrap();
    let output = call(
        Value::GpuTensor(input.clone()),
        &[Value::Num(8.0), Value::Num(0.0), Value::Num(8.0)],
        Some(1),
    )
    .unwrap();
    assert!(matches!(output, Value::OutputList(values) if values.len() == 1));
    assert_eq!(provider.integer_downloads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.other_downloads.load(Ordering::SeqCst), 0);
    assert!(input.allocation_lease().validate_live().is_ok());
}

#[test]
fn trace_modes_select_segment_powers_without_changing_the_frequency_grid() {
    let x = Value::Tensor(Tensor::new(vec![1.0, 1.0, 3.0, 3.0], vec![4, 1]).unwrap());
    let controls = [
        Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
        Value::Num(0.0),
        Value::Num(2.0),
        Value::Num(1.0),
    ];
    for (name, expected) in [("mean", 10.0), ("maxhold", 18.0), ("minhold", 2.0)] {
        let mut args = controls.to_vec();
        args.push(Value::String(name.into()));
        let (pxx, frequency) = output_pair(call(x.clone(), &args, Some(2)).unwrap());
        assert_eq!(frequency, vec![0.0, 0.5]);
        assert!((pxx[0] - expected).abs() < 1e-12, "{name}: {pxx:?}");
        assert!(pxx[1].abs() < 1e-12);
    }
}

#[test]
fn trace_mean_does_not_overflow_a_representable_average() {
    let first = f64::MAX.sqrt() * 0.5;
    let x = Value::Tensor(Tensor::new(vec![first, first], vec![2, 1]).unwrap());
    let args = [
        Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        Value::Num(0.0),
        Value::Num(1.0),
        Value::Num(1.0),
        Value::String("mean".into()),
    ];
    let (pxx, _) = output_pair(call(x, &args, Some(2)).unwrap());
    assert!(pxx[0].is_finite() && pxx[0] > f64::MAX * 0.2);
}

#[test]
fn trace_holds_include_nan_instead_of_hiding_a_missing_segment() {
    for trace in [WelchTrace::MaxHold, WelchTrace::MinHold] {
        assert!(combine_trace(2.0, f64::NAN, 1, trace).is_nan());
        assert!(combine_trace(f64::NAN, 2.0, 1, trace).is_nan());
    }
}

#[test]
fn shorter_dft_folds_the_segment_modulo_its_length() {
    let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap());
    let args = [
        Value::Tensor(Tensor::new(vec![1.0; 4], vec![4, 1]).unwrap()),
        Value::Num(0.0),
        Value::Num(2.0),
        Value::Num(1.0),
    ];
    let (pxx, f) = output_pair(call(x, &args, Some(2)).unwrap());
    assert_eq!(f, vec![0.0, 0.5]);
    assert!((pxx[0] - 25.0).abs() < 1e-12);
    assert!((pxx[1] - 1.0).abs() < 1e-12);
}

#[test]
#[cfg(feature = "plot-core")]
fn no_output_welch_publishes_a_line_in_the_current_axes() {
    let _guard = crate::builtins::plotting::tests::lock_plot_registry();
    let _ = crate::builtins::plotting::clear_figure(None);
    let axes = crate::builtins::plotting::state::current_axes_state();
    let before = crate::builtins::plotting::state::plot_child_handles_for_axes(
        axes.handle,
        axes.active_index,
    )
    .len();
    let x = Value::Tensor(Tensor::new(vec![1.0; 8], vec![8, 1]).unwrap());
    let output = call(
        x,
        &[Value::Num(4.0), Value::Num(0.0), Value::Num(4.0)],
        Some(0),
    )
    .unwrap();
    assert!(matches!(output, Value::OutputList(values) if values.is_empty()));
    let after = crate::builtins::plotting::state::plot_child_handles_for_axes(
        axes.handle,
        axes.active_index,
    )
    .len();
    assert_eq!(after, before + 1);
}

#[test]
#[cfg(feature = "plot-core")]
fn presentation_context_alone_publishes_and_discards_numeric_outputs() {
    let _guard = crate::builtins::plotting::tests::lock_plot_registry();
    let _ = crate::builtins::plotting::clear_figure(None);
    let axes = crate::builtins::plotting::state::current_axes_state();
    let before = crate::builtins::plotting::state::plot_child_handles_for_axes(
        axes.handle,
        axes.active_index,
    )
    .len();
    let _presentation = crate::output_context::push_output_count(0).expect("output-count scope");
    assert_eq!(crate::output_count::current_output_count(), None);
    let signal = Value::Tensor(Tensor::new(vec![1.0; 8], vec![8, 1]).unwrap());
    let output = call(
        signal,
        &[Value::Num(4.0), Value::Num(0.0), Value::Num(4.0)],
        None,
    )
    .unwrap();
    assert!(matches!(output, Value::OutputList(values) if values.is_empty()));
    let after = crate::builtins::plotting::state::plot_child_handles_for_axes(
        axes.handle,
        axes.active_index,
    )
    .len();
    assert_eq!(after, before + 1);
}

#[test]
#[cfg(not(feature = "plot-core"))]
fn no_plot_core_rejects_invalid_parent_before_missing_presentation() {
    let x = Value::Tensor(Tensor::new(vec![1.0; 8], vec![8, 1]).unwrap());
    for target in [Value::Num(0.0), Value::String("axes".into())] {
        let error = call(
            x.clone(),
            &[Value::String("Parent".into()), target],
            Some(1),
        )
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:pwelch:InvalidParent"));
    }
    let error = call(
        x,
        &[Value::String("Parent".into()), Value::Num(1.0)],
        Some(1),
    )
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        crate::plotting_hooks::unavailable_presentation().identifier()
    );
}

#[test]
#[cfg(not(feature = "plot-core"))]
fn presentation_context_alone_requires_plotting_capability() {
    let _presentation = crate::output_context::push_output_count(0).expect("output-count scope");
    assert_eq!(crate::output_count::current_output_count(), None);
    let x = Value::Tensor(Tensor::new(vec![1.0; 8], vec![8, 1]).unwrap());
    let error = call(
        x,
        &[Value::Num(4.0), Value::Num(0.0), Value::Num(4.0)],
        None,
    )
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        crate::plotting_hooks::unavailable_presentation().identifier()
    );
}

#[test]
#[cfg(feature = "plot-core")]
fn parent_name_value_uses_existing_axes_identity_only() {
    let _guard = crate::builtins::plotting::tests::lock_plot_registry();
    let axes = crate::builtins::plotting::state::current_axes_state();
    let target =
        crate::builtins::plotting::state::encode_axes_handle(axes.handle, axes.active_index);
    let x = Value::Tensor(Tensor::new(vec![1.0; 8], vec![8, 1]).unwrap());
    let output = call(
        x.clone(),
        &[
            Value::Num(4.0),
            Value::Num(0.0),
            Value::Num(4.0),
            Value::String("Parent".into()),
            Value::Num(target),
        ],
        Some(2),
    )
    .unwrap();
    assert!(matches!(output, Value::OutputList(values) if values.len() == 2));
    let error = call(
        x,
        &[Value::String("Parent".into()), Value::Num(0.0)],
        Some(1),
    )
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:pwelch:InvalidParent"));
}

#[test]
fn descriptor_is_registered() {
    let builtin = runmat_builtins::builtin_catalog_entry_by_name(BUILTIN_NAME)
        .expect("canonical pwelch identity");
    assert!(builtin_function_by_name(BUILTIN_NAME).is_none());
    assert_eq!(builtin.bindings.len(), 1);
    let bindings = crate::builtin::runtime_builtin_bindings_by_name(BUILTIN_NAME);
    assert_eq!(bindings.len(), 1);
    assert_eq!(
        bindings[0].identity,
        builtin.binding_identity(&builtin.bindings[0])
    );
    let descriptor = builtin.descriptor;
    assert!(descriptor
        .signatures
        .iter()
        .any(|sig| sig.label == "[pxx, f] = pwelch(x, window, noverlap, nfft, fs, options)"));
}

#[test]
fn pwelch_constant_signal_has_dc_peak() {
    let x = Tensor::new(vec![1.0; 16], vec![1, 16]).unwrap();
    let window = Tensor::new(vec![1.0; 8], vec![1, 8]).unwrap();
    let out = call(
        Value::Tensor(x),
        &[
            Value::Tensor(window),
            Value::Num(4.0),
            Value::Num(8.0),
            Value::Num(8.0),
        ],
        Some(2),
    )
    .unwrap();
    let (pxx, f) = output_pair(out);
    assert_eq!(pxx.len(), 5);
    assert_eq!(f.len(), 5);
    assert!(pxx[0] > 0.9);
    assert!(pxx[1..].iter().all(|value| value.abs() < 1e-12));
    assert_eq!(f[0], 0.0);
    assert_eq!(f[4], 4.0);
}

#[test]
fn pwelch_detects_sinusoid_bin() {
    let x = (0..32)
        .map(|idx| (2.0 * std::f64::consts::PI * 4.0 * idx as f64 / 32.0).sin())
        .collect::<Vec<_>>();
    let out = call(
        Value::Tensor(Tensor::new(x, vec![1, 32]).unwrap()),
        &[
            Value::Num(32.0),
            Value::Num(0.0),
            Value::Num(32.0),
            Value::Num(32.0),
        ],
        Some(2),
    )
    .unwrap();
    let (pxx, f) = output_pair(out);
    let (peak_idx, _) = pxx
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    assert_eq!(peak_idx, 4);
    assert_eq!(f[peak_idx], 4.0);
}

#[test]
fn pwelch_reads_typed_integer_vector_storage_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = call(
        Value::Tensor(integer_tensor(
            vec![0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1],
            vec![1, 16],
        )),
        &[
            Value::Num(16.0),
            Value::Num(0.0),
            Value::Num(16.0),
            Value::Num(16.0),
        ],
        Some(2),
    )
    .unwrap();
    let (pxx, f) = output_pair(out);
    let (peak_idx, _) = pxx
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    assert_eq!(peak_idx, 4);
    assert_eq!(f[peak_idx], 4.0);
}

#[test]
fn pwelch_scalar_window_reads_typed_integer_storage_length_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let window =
        Tensor::new_integer(IntegerStorage::I16(vec![8]), vec![1, 1]).expect("window length");

    let out = call(
        Value::Tensor(Tensor::new(vec![1.0; 16], vec![1, 16]).unwrap()),
        &[
            Value::Tensor(window),
            Value::Num(0.0),
            Value::Num(8.0),
            Value::Num(8.0),
        ],
        Some(2),
    )
    .unwrap();
    let (pxx, f) = output_pair(out);
    assert_eq!(pxx.len(), 5);
    assert_eq!(f.len(), 5);
}

#[test]
fn pwelch_scalar_detector_reads_typed_complex_integer_storage_without_mirror() {
    let storage = runmat_value::IntegerComplexStorage::new(
        IntegerStorage::I16(vec![8]),
        IntegerStorage::I16(vec![0]),
    )
    .expect("complex integer storage");
    let scalar = runmat_value::ComplexTensor::new_integer(storage, vec![1, 1]).expect("scalar");

    assert!(is_scalar_numeric(&Value::ComplexTensor(scalar)));
}

#[test]
fn pwelch_twosided_and_centered_ranges() {
    let data = (0..8)
        .map(|idx| (idx as f64, if idx % 2 == 0 { 0.0 } else { 0.25 }))
        .collect::<Vec<_>>();
    let x = runmat_value::ComplexTensor::new(data, vec![1, 8]).unwrap();
    let two = call(
        Value::ComplexTensor(x.clone()),
        &[
            Value::Num(8.0),
            Value::Num(0.0),
            Value::Num(8.0),
            Value::from("twosided"),
        ],
        Some(2),
    )
    .unwrap();
    let (pxx, f) = output_pair(two);
    assert_eq!(pxx.len(), 8);
    assert_eq!(f.len(), 8);

    let centered = call(
        Value::ComplexTensor(x),
        &[
            Value::Num(8.0),
            Value::Num(0.0),
            Value::Num(8.0),
            Value::from("centered"),
        ],
        Some(2),
    )
    .unwrap();
    let (_, f) = output_pair(centered);
    assert!((f[0] + 3.0 * std::f64::consts::PI / 4.0).abs() < 1e-12);
    assert_eq!(f[3], 0.0);
    assert_eq!(f[7], std::f64::consts::PI);
}

#[test]
fn pwelch_centered_odd_nfft_uses_fftshift_order() {
    let x = runmat_value::ComplexTensor::new(
        (0..5).map(|idx| (idx as f64, 0.25)).collect(),
        vec![1, 5],
    )
    .unwrap();
    let out = call(
        Value::ComplexTensor(x),
        &[
            Value::Num(5.0),
            Value::Num(0.0),
            Value::Num(5.0),
            Value::from("centered"),
        ],
        Some(2),
    )
    .unwrap();
    let (_, f) = output_pair(out);
    assert_eq!(f.len(), 5);
    assert!((f[0] + 4.0 * std::f64::consts::PI / 5.0).abs() < 1e-12);
    assert_eq!(f[2], 0.0);
    assert!((f[4] - 4.0 * std::f64::consts::PI / 5.0).abs() < 1e-12);
}

#[test]
fn pwelch_accepts_matrix_columns() {
    let data = vec![
        1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0,
    ];
    let x = Tensor::new(data, vec![8, 2]).unwrap();
    let out = call(
        Value::Tensor(x),
        &[Value::Num(8.0), Value::Num(0.0), Value::Num(8.0)],
        Some(2),
    )
    .unwrap();
    let (pxx, shape, f) = output_pair_with_pxx_shape(out);
    assert_eq!(shape, vec![5, 2]);
    assert_eq!(pxx.len(), 10);
    assert_eq!(f.len(), 5);
    assert!(pxx[0] > pxx[5]);
}

#[test]
fn pwelch_preserves_single_output_class() {
    let signal = Tensor::from_f32(vec![1.0; 8], vec![1, 8]).expect("single signal");
    let output = call(
        Value::Tensor(signal),
        &[Value::Num(8.0), Value::Num(0.0), Value::Num(8.0)],
        Some(2),
    )
    .expect("pwelch");
    let Value::OutputList(outputs) = output else {
        panic!("expected two outputs")
    };
    for output in outputs.try_into_owned().expect("unique output list") {
        let Value::Tensor(output) = output else {
            panic!("expected tensor output")
        };
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
    }
}

#[test]
fn pwelch_reads_typed_integer_matrix_storage_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let data = vec![
        1, 1, 1, 1, 0, 0, 0, 0, //
        0, 1, 0, -1, 0, 1, 0, -1,
    ];
    let out = call(
        Value::Tensor(integer_tensor(data, vec![8, 2])),
        &[Value::Num(8.0), Value::Num(0.0), Value::Num(8.0)],
        Some(2),
    )
    .unwrap();
    let (pxx, shape, f) = output_pair_with_pxx_shape(out);
    assert_eq!(shape, vec![5, 2]);
    assert_eq!(pxx.len(), 10);
    assert_eq!(f.len(), 5);
    assert!(pxx[0] > pxx[5]);
}

#[test]
fn pwelch_integer_extensions_are_gated_and_inexact_values_reject() {
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let signal =
        Tensor::new_integer(IntegerStorage::I16(vec![1; 8]), vec![1, 8]).expect("integer signal");
    let error = call(Value::Tensor(signal), &[], Some(1))
        .expect_err("strict mode rejects integer signal extension");
    assert_eq!(
        error.identifier(),
        PWELCH_INTEGER_SIGNAL_EXTENSION.error_identifier
    );
    drop(strict);

    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let window =
        Tensor::new_integer(IntegerStorage::I16(vec![8]), vec![1, 1]).expect("integer window");
    let error = call(
        Value::Tensor(Tensor::new(vec![1.0; 8], vec![1, 8]).unwrap()),
        &[Value::Tensor(window)],
        Some(1),
    )
    .expect_err("strict mode rejects integer control extension");
    assert_eq!(
        error.identifier(),
        PWELCH_INTEGER_CONTROL_EXTENSION.error_identifier
    );
    drop(strict);

    let extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let wide = Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1; 8]), vec![1, 8])
        .expect("wide integer signal");
    let error = call(Value::Tensor(wide), &[], Some(1))
        .expect_err("inexact integer signal rejects at floating boundary");
    assert!(error.message().contains("exactly representable as double"));
    drop(extensions);
}

#[test]
fn pwelch_logical_input_is_a_gated_extension() {
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = call(Value::Bool(true), &[], Some(1))
        .expect_err("strict mode rejects logical signal extension");
    assert_eq!(
        error.identifier(),
        PWELCH_LOGICAL_INPUT_EXTENSION.error_identifier
    );
    drop(strict);

    let extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let output = call(
        Value::Tensor(Tensor::new(vec![1.0; 8], vec![1, 8]).unwrap()),
        &[Value::Bool(true)],
        Some(1),
    );
    assert!(
        output.is_ok(),
        "RunMat mode retains logical control admission"
    );
    drop(extensions);
}

#[test]
fn pwelch_automatic_integer_residency_restores_floating_outputs_to_exact_owner() {
    test_support::with_test_provider(|provider| {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let signal = Tensor::new_integer(
            IntegerStorage::I16(vec![0, 1, 0, -1, 0, 1, 0, -1]),
            vec![1, 8],
        )
        .expect("integer signal");
        let input = gpu_helpers::upload_tensor(provider, &signal).expect("upload");
        let output = call(
            Value::GpuTensor(input.clone()),
            &[Value::Num(8.0), Value::Num(0.0), Value::Num(8.0)],
            Some(2),
        )
        .expect("pwelch");
        let Value::OutputList(outputs) = output else {
            panic!("expected two resident outputs")
        };
        assert_eq!(outputs.len(), 2);
        for output in &outputs {
            let Value::GpuTensor(handle) = output else {
                panic!("expected resident output")
            };
            assert!(
                runmat_accelerate_api::provider_for_handle(handle).is_some_and(|owner| {
                    runmat_accelerate_api::same_provider_instance(owner, provider)
                })
            );
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(handle),
                None,
                "Welch outputs are floating"
            );
            assert_eq!(
                runmat_accelerate_api::handle_precision(handle),
                Some(runmat_accelerate_api::ProviderPrecision::F64)
            );
        }
        let gathered = block_on(gpu_helpers::gather_value_async(&outputs[0])).expect("gather");
        let Value::Tensor(pxx) = gathered else {
            panic!("expected floating PSD")
        };
        assert_eq!(pxx.shape(), vec![5, 1]);
    });
}

#[test]
fn resident_scalar_controls_are_decoded_as_structural_values_once() {
    test_support::with_test_provider(|provider| {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let x = Tensor::new(vec![1.0; 16], vec![1, 16]).unwrap();
        let window = Tensor::new_integer(IntegerStorage::U16(vec![8]), vec![1, 1]).unwrap();
        let overlap = Tensor::new_integer(IntegerStorage::U16(vec![4]), vec![1, 1]).unwrap();
        let nfft = Tensor::new_integer(IntegerStorage::U16(vec![16]), vec![1, 1]).unwrap();
        let fs = Tensor::new(vec![100.0], vec![1, 1]).unwrap();
        let resident = [window, overlap, nfft, fs]
            .iter()
            .map(|tensor| gpu_helpers::upload_tensor(provider, tensor).expect("upload control"))
            .map(Value::GpuTensor)
            .collect::<Vec<_>>();
        let output = call(Value::Tensor(x), &resident, Some(2)).expect("Welch controls");
        let (pxx, f) = output_pair(output);
        assert_eq!(pxx.len(), 9);
        assert_eq!(f.len(), 9);
        assert!((f[1] - 6.25).abs() < 1e-12);
    });
}

#[test]
fn known_invalid_host_control_precedes_resident_control_access() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let window = Tensor::new_integer(IntegerStorage::U16(vec![8]), vec![1, 1]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &window).expect("upload control");
        let x = Tensor::new(vec![1.0; 16], vec![1, 16]).unwrap();
        let error = call(
            Value::Tensor(x),
            &[Value::GpuTensor(handle), Value::Num(-1.0)],
            Some(1),
        )
        .expect_err("invalid host overlap");
        assert_eq!(
            error.identifier.as_deref(),
            Some("RunMat:pwelch:InvalidOverlap")
        );
    });
}

#[test]
fn inexact_wide_host_window_rejects_before_length_allocation() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let signal = Tensor::new(vec![1.0; 8], vec![1, 8]).unwrap();
    let window =
        Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]).unwrap();
    let error = call(Value::Tensor(signal), &[Value::Tensor(window)], Some(1))
        .expect_err("inexact integer length");
    assert_eq!(
        error.identifier.as_deref(),
        Some("RunMat:pwelch:InvalidWindow")
    );
    assert!(error.message().contains("exactly representable"));
}

#[test]
fn pwelch_accepts_explicit_frequency_vector() {
    let x = Tensor::new(vec![1.0; 4], vec![1, 4]).unwrap();
    let window = Tensor::new(vec![1.0; 4], vec![1, 4]).unwrap();
    let frequencies = Tensor::new(vec![0.0, std::f64::consts::PI], vec![2, 1]).unwrap();
    let out = call(
        Value::Tensor(x),
        &[
            Value::Tensor(window),
            Value::Num(0.0),
            Value::Tensor(frequencies),
        ],
        Some(2),
    )
    .unwrap();
    let (pxx, f) = output_pair(out);
    assert_eq!(f, vec![0.0, std::f64::consts::PI]);
    assert_eq!(pxx.len(), 2);
    assert!(pxx[0] > 0.5);
    assert!(pxx[1].abs() < 1e-12);
}

#[test]
fn pwelch_short_nfft_folds_windowed_samples_modulo_transform_length() {
    let x = Tensor::new(vec![1.0; 16], vec![1, 16]).unwrap();
    let out = call(
        Value::Tensor(x),
        &[
            Value::Tensor(Tensor::new(vec![1.0; 8], vec![1, 8]).unwrap()),
            Value::Num(0.0),
            Value::Num(4.0),
            Value::Num(1.0),
        ],
        Some(2),
    )
    .unwrap();
    let (pxx, frequency) = output_pair(out);
    assert_eq!(frequency, vec![0.0, 0.25, 0.5]);
    assert!((pxx[0] - 8.0).abs() < 1e-12);
    assert!(pxx[1..].iter().all(|power| power.abs() < 1e-12));
}

#[test]
fn pwelch_power_uses_window_equivalent_noise_bandwidth() {
    let x = Tensor::new(vec![1.0; 8], vec![1, 8]).unwrap();
    let window = Tensor::new(vec![1.0; 8], vec![1, 8]).unwrap();
    let psd = call(
        Value::Tensor(x.clone()),
        &[
            Value::Tensor(window.clone()),
            Value::Num(0.0),
            Value::Num(8.0),
            Value::Num(8.0),
            Value::from("psd"),
        ],
        Some(2),
    )
    .unwrap();
    let power = call(
        Value::Tensor(x),
        &[
            Value::Tensor(window),
            Value::Num(0.0),
            Value::Num(8.0),
            Value::Num(8.0),
            Value::from("power"),
        ],
        Some(2),
    )
    .unwrap();
    let (psd, _) = output_pair(psd);
    let (power, _) = output_pair(power);
    // For a rectangular eight-sample window at Fs=8, ENBW is 1 Hz.
    assert!((power[0] / psd[0] - 1.0).abs() < 1e-12);

    let signal = Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap());
    let weighted = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
    let base = [
        weighted.clone(),
        Value::Num(0.0),
        Value::Num(2.0),
        Value::Num(1.0),
    ];
    let (density, _) = output_pair(call(signal.clone(), &base, Some(2)).unwrap());
    let (power, _) = output_pair(
        call(
            signal,
            &[
                weighted,
                Value::Num(0.0),
                Value::Num(2.0),
                Value::Num(1.0),
                Value::from("power"),
            ],
            Some(2),
        )
        .unwrap(),
    );
    assert!((density[0] - 9.0 / 5.0).abs() < 1e-12);
    assert!((power[0] - 1.0).abs() < 1e-12);
}

#[test]
fn explicit_empty_sample_rate_is_one_hertz_but_omitted_is_angular() {
    let signal = Value::Tensor(Tensor::new(vec![1.0; 8], vec![1, 8]).unwrap());
    let window = Value::Tensor(Tensor::new(vec![1.0; 8], vec![1, 8]).unwrap());
    let common = [window.clone(), Value::Num(0.0), Value::Num(8.0)];
    let (_, angular) = output_pair(call(signal.clone(), &common, Some(2)).unwrap());
    let (_, hertz) = output_pair(
        call(
            signal,
            &[
                window,
                Value::Num(0.0),
                Value::Num(8.0),
                Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
            ],
            Some(2),
        )
        .unwrap(),
    );
    assert!((angular[1] - std::f64::consts::PI / 4.0).abs() < 1e-12);
    assert!((hertz[1] - 0.125).abs() < 1e-12);
}

#[test]
fn finite_tiny_and_huge_windows_preserve_normalized_psd() {
    for amplitude in [1e-200, 1e200] {
        let result = call(
            Value::Tensor(Tensor::new(vec![1.0; 2], vec![1, 2]).unwrap()),
            &[
                Value::Tensor(Tensor::new(vec![amplitude; 2], vec![1, 2]).unwrap()),
                Value::Num(0.0),
                Value::Num(2.0),
                Value::Num(1.0),
            ],
            Some(2),
        )
        .unwrap();
        let (pxx, frequency) = output_pair(result);
        assert_eq!(frequency, vec![0.0, 0.5]);
        assert!((pxx[0] - 2.0).abs() < 1e-12);
        assert!(pxx[1].abs() < 1e-12);
    }
    let scaled = call(
        Value::Num(1e200),
        &[
            Value::Num(1.0),
            Value::Num(0.0),
            Value::Num(1.0),
            Value::Num(1e200),
        ],
        Some(2),
    )
    .unwrap();
    let (scaled_psd, _) = output_pair(scaled);
    assert!((scaled_psd[0] / 1e200 - 1.0).abs() < 1e-12);

    let extreme_fs = call(
        Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
        &[
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
            Value::Num(0.0),
            Value::Num(2.0),
            Value::Num(f64::MAX),
            Value::from("twosided"),
        ],
        Some(2),
    )
    .unwrap();
    let (extreme_psd, _) = output_pair(extreme_fs);
    let expected = 2.0 / f64::MAX;
    assert!(extreme_psd[0] > 0.0);
    assert!((extreme_psd[0] / expected - 1.0).abs() < 1e-12);
    let extreme_frequency = futures::executor::block_on(frequency_vector(
        4,
        Some(f64::MAX),
        FrequencyRange::Twosided,
        &mut SignalCooperation::new(None, BUILTIN_NAME),
    ))
    .unwrap();
    assert!(extreme_frequency.iter().all(|value| value.is_finite()));
    assert_eq!(extreme_frequency[2], f64::MAX / 2.0);
}

#[test]
fn spectral_output_geometry_rejects_product_overflow() {
    let error = checked_output_elements(usize::MAX, 2).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:pwelch:Internal"));
    assert!(error.message().contains("dimensions overflow"));
}

#[cfg(feature = "wgpu")]
#[test]
fn pwelch_wgpu_keeps_uniform_outputs_resident_and_matches_cpu() {
    let _guard = crate::builtins::common::test_support::accel_test_lock();
    let Some(provider) =
        runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider().expect("wgpu provider")
    else {
        return;
    };
    let _guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));

    let rows = 64usize;
    let mut data = Vec::with_capacity(rows * 2);
    data.extend((0..rows).map(|idx| (2.0 * std::f64::consts::PI * 4.0 * idx as f64 / 32.0).sin()));
    data.extend(std::iter::repeat_n(1.0, rows));
    let cpu_x = Tensor::new(data.clone(), vec![rows, 2]).unwrap();
    let cpu = call(
        Value::Tensor(cpu_x),
        &[
            Value::Num(16.0),
            Value::Num(8.0),
            Value::Num(32.0),
            Value::Num(32.0),
        ],
        Some(2),
    )
    .expect("pwelch cpu");
    let (cpu_pxx, cpu_f) = output_pair(cpu);

    let shape = [rows, 2usize];
    let handle = provider
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &data,
            shape: &shape,
        })
        .expect("upload");
    let gpu = call(
        Value::GpuTensor(handle.clone()),
        &[
            Value::Num(16.0),
            Value::Num(8.0),
            Value::Num(32.0),
            Value::Num(32.0),
        ],
        Some(2),
    )
    .expect("pwelch gpu");
    let Value::OutputList(values) = gpu else {
        panic!("expected output list");
    };
    for value in &values {
        assert!(matches!(value, Value::GpuTensor(_)));
    }
    let gpu_pxx =
        crate::builtins::common::test_support::gather(values[0].clone()).expect("gather pxx");
    let gpu_f = crate::builtins::common::test_support::gather(values[1].clone()).expect("gather f");

    assert_eq!(gpu_pxx.shape(), vec![17, 2]);
    assert_eq!(gpu_f.shape(), vec![17, 1]);
    assert_eq!(gpu_f.materialize_f64(), cpu_f);
    for (actual, expected) in gpu_pxx.materialize_f64().iter().zip(cpu_pxx.iter()) {
        assert!(
            (actual - expected).abs() <= 1.0e-5,
            "actual={actual} expected={expected}"
        );
    }
    let tone = runmat_value::ComplexTensor::new(
        (0..8)
            .map(|index| {
                let angle = 2.0 * std::f64::consts::PI * index as f64 / 8.0;
                (angle.cos(), angle.sin())
            })
            .collect(),
        vec![8, 1],
    )
    .unwrap();
    let controls = [
        Value::Num(8.0),
        Value::Num(0.0),
        Value::Num(8.0),
        Value::Num(8.0),
        Value::from("centered"),
    ];
    let (host_centered, host_f) =
        output_pair(call(Value::ComplexTensor(tone.clone()), &controls, Some(2)).unwrap());
    let resident_tone = gpu_helpers::upload_complex_tensor(provider, &tone).unwrap();
    let Value::OutputList(resident_centered) =
        call(Value::GpuTensor(resident_tone), &controls, Some(2)).unwrap()
    else {
        panic!("centered Welch must return two outputs");
    };
    let resident_power =
        crate::builtins::common::test_support::gather(resident_centered[0].clone()).unwrap();
    let resident_f =
        crate::builtins::common::test_support::gather(resident_centered[1].clone()).unwrap();
    assert_eq!(resident_f.materialize_f64(), host_f);
    assert_eq!(host_f[0], -3.0);
    assert_eq!(host_f[7], 4.0);
    for (actual, expected) in resident_power
        .materialize_f64()
        .iter()
        .zip(host_centered.iter())
    {
        assert!((actual - expected).abs() < 1e-5);
    }
    assert!(host_centered[4] > host_centered[3]);
    let extreme_input = Tensor::new(vec![1e200], vec![1, 1]).unwrap();
    let extreme_handle = gpu_helpers::upload_tensor(provider, &extreme_input).unwrap();
    let extreme = call(
        Value::GpuTensor(extreme_handle),
        &[
            Value::Num(1.0),
            Value::Num(0.0),
            Value::Num(1.0),
            Value::Num(1e200),
        ],
        Some(1),
    )
    .unwrap();
    let Value::OutputList(extreme) = extreme else {
        panic!("one requested output must retain its output list");
    };
    let extreme = crate::builtins::common::test_support::gather(extreme[0].clone()).unwrap();
    assert!((extreme.materialize_f64()[0] / 1e200 - 1.0).abs() < 1e-5);
    handle.release().ok();
}

#[test]
fn default_window_uses_eight_overlapped_sections() {
    let mut cooperation = SignalCooperation::new(None, BUILTIN_NAME);
    assert_eq!(
        block_on(default_window(900, &mut cooperation))
            .unwrap()
            .len(),
        200
    );
    assert_eq!(
        block_on(default_window(1, &mut cooperation)).unwrap().len(),
        1
    );
}

#[test]
fn default_fft_grid_rounds_window_length_to_next_power_of_two() {
    let signal = Value::Tensor(Tensor::new(vec![1.0; 300], vec![1, 300]).unwrap());
    let (_, frequencies) = output_pair(call(signal, &[Value::Num(300.0)], Some(2)).unwrap());
    assert_eq!(frequencies.len(), 257);
    assert!((frequencies[1] - 2.0 * std::f64::consts::PI / 512.0).abs() < 1e-12);
}

#[test]
fn pwelch_rejects_invalid_overlap() {
    let x = Tensor::new(vec![1.0; 8], vec![1, 8]).unwrap();
    let err = call(
        Value::Tensor(x),
        &[Value::Num(8.0), Value::Num(8.0)],
        Some(2),
    )
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:pwelch:InvalidOverlap"));
}

mod cooperation_tests {
    use super::*;
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
    use futures::task::noop_waker;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::cell::Cell;
    use std::future::Future;
    use std::num::NonZeroU64;
    use std::pin::Pin;
    use std::rc::Rc;
    use std::task::{Context, Poll};

    struct PendingTurn(Rc<Cell<usize>>);
    impl Future for PendingTurn {
        type Output = Result<(), ExecutionServiceError>;
        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Pending
        }
    }
    impl Drop for PendingTurn {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    #[test]
    fn pending_window_generation_drop_and_cancel_never_publish() {
        for cancel in [false, true] {
            let source = CancellationSource::new();
            let dropped = Rc::new(Cell::new(0));
            let count = dropped.clone();
            let budget = CpuCooperationBudget::with_test_turn(
                CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(4).unwrap()),
                source.token(),
                move || Box::pin(PendingTurn(count.clone())),
            );
            let mut cooperation = SignalCooperation::with_budget(budget, BUILTIN_NAME);
            let mut future = Box::pin(symmetric_window_values(
                8193,
                hamming_coefficient,
                &mut cooperation,
            ));
            let waker = noop_waker();
            let mut context = Context::from_waker(&waker);
            assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
            if cancel {
                source.cancel(CancellationReason::User);
                let Poll::Ready(Err(error)) = future.as_mut().poll(&mut context) else {
                    panic!("cancelled window generation must not publish a window");
                };
                assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
                assert!(matches!(
                    error
                        .source
                        .as_deref()
                        .and_then(|cause| cause.downcast_ref::<ExecutionServiceError>()),
                    Some(ExecutionServiceError::Cancelled)
                ));
            }
            drop(future);
            assert_eq!(dropped.get(), 1);
        }
    }

    #[test]
    fn invocation_service_failure_keeps_cause_and_ready_value_matches_direct_call() {
        let signal = Value::Tensor(Tensor::new(vec![1.0; 16], vec![1, 16]).unwrap());
        let args = [Value::Num(8.0), Value::Num(4.0), Value::Num(8.0)];
        let expected = call(signal.clone(), &args, None).unwrap();
        let source = CancellationSource::new();
        let ready_budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            || Box::pin(async { Ok(()) }),
        );
        let actual = block_on(evaluate_with_cooperation(
            signal.clone(),
            crate::call::FunctionArgumentView::from_slice(&args),
            &mut SignalCooperation::with_budget(ready_budget, BUILTIN_NAME),
        ))
        .unwrap();
        let (Value::Tensor(expected), Value::Tensor(actual)) = (expected, actual) else {
            panic!("expected real PSD tensors");
        };
        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(actual.materialize_f64(), expected.materialize_f64());

        let failure_budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            || Box::pin(async { Err(ExecutionServiceError::Infrastructure("turn failed".into())) }),
        );
        let error = block_on(evaluate_with_cooperation(
            signal,
            crate::call::FunctionArgumentView::from_slice(&args),
            &mut SignalCooperation::with_budget(failure_budget, BUILTIN_NAME),
        ))
        .unwrap_err();
        assert_eq!(error.context.builtin.as_deref(), Some(BUILTIN_NAME));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(matches!(
            error
                .source
                .as_deref()
                .and_then(|cause| cause.downcast_ref::<ExecutionServiceError>()),
            Some(ExecutionServiceError::Infrastructure(_))
        ));
    }
}

#[path = "tests/cooperation_resident.rs"]
mod cooperation_resident;
