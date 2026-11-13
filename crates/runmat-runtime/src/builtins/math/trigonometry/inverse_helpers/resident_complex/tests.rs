use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::AccelProvider;
use runmat_value::{ComplexTensor, NumericDType};

const IDENTITIES: &[&str] = &["asin", "acos", "acosh", "atanh", "asinh", "atan", "tanh"];

fn check(provider: &'static dyn AccelProvider, single: bool) {
    // Already-complex inputs must not require the real-to-complex GPU extension.
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let input = if single {
        ComplexTensor::from_f32(vec![(0.25, 0.5), (-0.75, -0.25)], vec![1, 2, 1]).unwrap()
    } else {
        ComplexTensor::new(vec![(0.25, 0.5), (-0.75, -0.25)], vec![1, 2, 1]).unwrap()
    };
    let handle = gpu_helpers::upload_complex_tensor(provider, &input).unwrap();
    for name in IDENTITIES {
        let expected = block_on(crate::dispatcher::call_builtin_async_with_outputs(
            name,
            &[Value::ComplexTensor(input.clone())],
            1,
        ))
        .unwrap();
        let result = block_on(crate::dispatcher::call_builtin_async_with_outputs(
            name,
            &[Value::GpuTensor(handle.clone())],
            1,
        ))
        .unwrap();
        let Value::GpuTensor(output_handle) = result.value() else {
            panic!("{name}: owner restoration required")
        };
        assert_eq!(output_handle.device_id(), handle.device_id());
        assert!(!gpu_helpers::same_gpu_handle(output_handle, &handle));
        let actual = block_on(gpu_helpers::gather_value_async(result.value())).unwrap();
        let (Value::ComplexTensor(actual), Value::ComplexTensor(expected)) =
            (actual, expected.value())
        else {
            panic!("{name}: complex array required");
        };
        assert_eq!(actual.shape(), input.shape(), "{name}");
        assert_eq!(
            actual.numeric_dtype(),
            if single {
                NumericDType::F32
            } else {
                NumericDType::F64
            }
        );
        let tolerance = if single { 1e-6 } else { 1e-12 };
        for (actual, expected) in actual
            .materialize_f64()
            .iter()
            .zip(expected.materialize_f64())
        {
            assert!(
                (actual.0 - expected.0).abs() < tolerance,
                "{name}: real component"
            );
            assert!(
                (actual.1 - expected.1).abs() < tolerance,
                "{name}: imaginary component"
            );
        }
        drop(result);
    }
    // Every invocation borrowed this handle; none may free it or rewrite its storage.
    let preserved = block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(
        handle.clone(),
    )))
    .unwrap();
    let Value::ComplexTensor(preserved) = preserved else {
        panic!("borrowed complex input was corrupted")
    };
    assert_eq!(preserved.materialize_f64(), input.materialize_f64());
    handle.release().unwrap();
}

#[test]
fn inverse_complex_resident_double_matches_host_without_real_promotion_extension() {
    test_support::with_test_provider(|provider| check(provider, false));
}

#[test]
fn inverse_complex_resident_single_matches_host_without_real_promotion_extension() {
    test_support::with_f32_test_provider(|provider| check(provider, true));
}
