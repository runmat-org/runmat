use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::AccelProvider;
use runmat_accelerate_api::{GpuHandleProvenance, HostIntegerDataView, HostIntegerTensorView};
use runmat_value::{
    CharArray, ComplexTensor, IntValue, IntegerStorage, LogicalArray, NumericDType,
};

fn call(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    block_on(gamma_builtin(value, rest))
}

fn approx_eq(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual} (tol {tolerance})"
    );
}

#[test]
fn rejects_excess_outputs() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let error = call(Value::Num(0.5), Vec::new()).expect_err("second output must reject");
    assert_eq!(error.identifier(), GAMMA_ERROR_TOO_MANY_OUTPUTS.identifier);
}

#[test]
fn scalar_real_values_cover_integer_half_and_negative_inputs() {
    match call(Value::Num(5.0), Vec::new()).unwrap() {
        Value::Num(value) => approx_eq(value, 24.0, 1e-12),
        other => panic!("expected scalar, got {other:?}"),
    }
    match call(Value::Num(0.5), Vec::new()).unwrap() {
        Value::Num(value) => approx_eq(value, PI.sqrt(), 1e-12),
        other => panic!("expected scalar, got {other:?}"),
    }
    match call(Value::Num(-0.5), Vec::new()).unwrap() {
        Value::Num(value) => approx_eq(value, -2.0 * PI.sqrt(), 1e-10),
        other => panic!("expected scalar, got {other:?}"),
    }
}

#[test]
fn poles_and_nonfinite_values_follow_real_contract() {
    assert!(matches!(
        call(Value::Num(0.0), Vec::new()).unwrap(),
        Value::Num(value) if value.is_infinite()
    ));
    assert!(matches!(
        call(Value::Num(f64::NAN), Vec::new()).unwrap(),
        Value::Num(value) if value.is_nan()
    ));
    assert!(matches!(
        call(Value::Num(f64::INFINITY), Vec::new()).unwrap(),
        Value::Num(value) if value.is_infinite()
    ));
}

#[test]
fn double_tensor_preserves_shape_and_values() {
    let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let output = call(Value::Tensor(input), Vec::new()).unwrap();
    match output {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape, vec![2, 2]);
            for (actual, expected) in tensor.materialize_f64().iter().zip([1.0, 1.0, 2.0, 6.0]) {
                approx_eq(*actual, expected, 1e-12);
            }
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn single_tensor_preserves_native_storage() {
    let input = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let output = call(Value::Tensor(input), Vec::new()).unwrap();
    match output {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
            assert_eq!(
                tensor.into_numeric_storage().unwrap(),
                NumericStorage::F32(vec![1.0, 1.0, 2.0, 6.0])
            );
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn documented_overflow_thresholds_follow_input_precision() {
    assert!(matches!(
        call(Value::Num(171.0), Vec::new()).unwrap(),
        Value::Num(value) if value.is_finite()
    ));
    assert!(matches!(
        call(Value::Num(172.0), Vec::new()).unwrap(),
        Value::Num(value) if value.is_infinite()
    ));

    let input = Tensor::from_f32(vec![35.0, 36.0], vec![1, 2]).unwrap();
    let Value::Tensor(output) = call(Value::Tensor(input), Vec::new()).unwrap() else {
        panic!("expected tensor output");
    };
    let values = output.materialize_f64();
    assert!(values[0].is_finite());
    assert!(values[1].is_infinite());
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
}

#[test]
fn integer_scalar_and_dense_integer_tensor_are_rejected() {
    let scalar = call(Value::Int(IntValue::I32(5)), Vec::new()).unwrap_err();
    assert_eq!(scalar.identifier(), GAMMA_ERROR_INVALID_INPUT.identifier);

    let tensor = Tensor::new_integer(IntegerStorage::U64(vec![1, u64::MAX]), vec![1, 2]).unwrap();
    let tensor = call(Value::Tensor(tensor), Vec::new()).unwrap_err();
    assert_eq!(tensor.identifier(), GAMMA_ERROR_INVALID_INPUT.identifier);
}

#[test]
fn logical_char_string_and_complex_inputs_are_rejected() {
    for value in [
        Value::Bool(true),
        Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap()),
        Value::CharArray(CharArray::new(vec!['A', 'Z'], 1, 2).unwrap()),
        Value::from("text"),
        Value::Complex(1.0, 1.0),
        Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 0.0)], vec![1, 1]).unwrap()),
    ] {
        let error = call(value, Vec::new()).unwrap_err();
        assert_eq!(error.identifier(), GAMMA_ERROR_INVALID_INPUT.identifier);
    }
}

#[test]
fn extra_like_and_other_arguments_are_rejected() {
    let digits = call(Value::Num(1.0), vec![Value::Num(2.0)]).unwrap_err();
    assert_eq!(digits.identifier(), GAMMA_ERROR_INVALID_ARGUMENT.identifier);
    let like = call(Value::Num(1.0), vec![Value::from("like"), Value::Num(0.0)]).unwrap_err();
    assert_eq!(like.identifier(), GAMMA_ERROR_INVALID_ARGUMENT.identifier);
}

#[test]
fn gpu_provider_roundtrip_preserves_residency() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let mut handle = gpu_helpers::upload_tensor(provider, &input).unwrap();
        runmat_accelerate_api::set_handle_provenance(&mut handle, GpuHandleProvenance::Explicit);
        let output = call(Value::GpuTensor(handle), Vec::new()).unwrap();
        let Value::GpuTensor(resident) = &output else {
            panic!("expected resident gamma output, got {output:?}");
        };
        assert_eq!(
            runmat_accelerate_api::handle_provenance(resident),
            Some(GpuHandleProvenance::Explicit)
        );
        let gathered = test_support::gather(output).unwrap();
        assert_eq!(gathered.shape, vec![2, 2]);
        assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
        for (actual, expected) in gathered.materialize_f64().iter().zip([1.0, 1.0, 2.0, 6.0]) {
            approx_eq(*actual, expected, 1e-12);
        }
    });
}

#[test]
fn integer_gpu_input_is_rejected_before_provider_gamma() {
    test_support::with_test_provider(|provider| {
        let values = [1u64, u64::MAX];
        let shape = [1usize, 2usize];
        let handle = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&values),
                shape: &shape,
            })
            .unwrap();
        let error = call(Value::GpuTensor(handle), Vec::new()).unwrap_err();
        assert_eq!(error.identifier(), GAMMA_ERROR_INVALID_INPUT.identifier);
    });
}

#[test]
fn logical_gpu_input_is_rejected_before_provider_gamma() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &input).unwrap();
        runmat_accelerate_api::set_handle_logical(&handle, true);
        let error = call(Value::GpuTensor(handle.clone()), Vec::new()).unwrap_err();
        assert_eq!(error.identifier(), GAMMA_ERROR_INVALID_INPUT.identifier);
        assert!(runmat_accelerate_api::provider_for_handle(&handle)
            .is_some_and(|owner| std::ptr::eq(owner, provider)));
    });
}

#[test]
fn complex_gpu_input_is_rejected_before_provider_gamma() {
    test_support::with_test_provider(|provider| {
        let input = ComplexTensor::new(vec![(1.0, 0.0)], vec![1, 1]).unwrap();
        let handle = gpu_helpers::upload_complex_tensor(provider, &input).unwrap();
        let error = call(Value::GpuTensor(handle), Vec::new()).unwrap_err();
        assert_eq!(error.identifier(), GAMMA_ERROR_INVALID_INPUT.identifier);
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_gamma_matches_host_for_real_inputs() {
    let _guard = test_support::accel_test_lock();
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };
    let input = Tensor::new(vec![0.25, 0.5, 1.0, 1.5], vec![2, 2]).unwrap();
    let expected = gamma_tensor(input.clone()).expect("host gamma");
    let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload gamma input");

    let output = call(Value::GpuTensor(handle), Vec::new()).expect("wgpu gamma");
    let actual = test_support::gather(output).expect("gather wgpu gamma");

    assert_eq!(actual.shape, expected.shape);
    let tolerance = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => 1e-8,
        runmat_accelerate_api::ProviderPrecision::F32 => 2e-4,
    };
    for (actual, expected) in actual
        .materialize_f64()
        .iter()
        .zip(expected.materialize_f64().iter())
    {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual} (tol {tolerance})"
        );
    }
}
