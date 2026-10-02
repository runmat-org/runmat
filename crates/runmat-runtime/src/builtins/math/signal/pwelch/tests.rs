use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::AccelProvider;
use runmat_builtins::builtin_function_by_name;
use runmat_value::IntegerStorage;

fn call(x: Value, rest: &[Value], outputs: Option<usize>) -> BuiltinResult<Value> {
    let _guard = outputs.map(|count| crate::output_count::push_output_count(Some(count)));
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
        pxx.shape.clone(),
        f.materialize_f64().clone(),
    )
}

fn integer_tensor(values: Vec<i16>, shape: Vec<usize>) -> Tensor {
    Tensor::new_integer(IntegerStorage::I16(values), shape).expect("typed integer tensor")
}

#[test]
fn descriptor_is_registered() {
    let builtin = builtin_function_by_name(BUILTIN_NAME).expect("pwelch builtin");
    let descriptor = builtin.descriptor.expect("descriptor");
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
    assert_eq!(f[0], -std::f64::consts::PI);
    assert_eq!(f[4], 0.0);
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
    for output in outputs {
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
        assert_eq!(pxx.shape, vec![5, 1]);
    });
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
fn pwelch_rejects_nfft_shorter_than_window() {
    let x = Tensor::new(vec![1.0; 16], vec![1, 16]).unwrap();
    let err = call(
        Value::Tensor(x),
        &[Value::Num(8.0), Value::Num(0.0), Value::Num(4.0)],
        Some(2),
    )
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:pwelch:InvalidNfft"));
}

#[test]
fn pwelch_power_scale_omits_density_bandwidth() {
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
    assert!((power[0] / psd[0] - 8.0).abs() < 1e-12);
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

    assert_eq!(gpu_pxx.shape, vec![17, 2]);
    assert_eq!(gpu_f.shape, vec![17, 1]);
    assert_eq!(gpu_f.materialize_f64(), cpu_f);
    for (actual, expected) in gpu_pxx.materialize_f64().iter().zip(cpu_pxx.iter()) {
        assert!(
            (actual - expected).abs() <= 1.0e-5,
            "actual={actual} expected={expected}"
        );
    }
    handle.release().ok();
}

#[test]
fn default_window_uses_eight_overlapped_sections() {
    assert_eq!(default_window(900).len(), 200);
    assert_eq!(default_window(1).len(), 1);
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
