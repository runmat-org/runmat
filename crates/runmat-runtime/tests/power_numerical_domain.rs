use futures::executor::block_on;
use runmat_value::{Tensor, Value};

#[path = "support/retained_provider.rs"]
mod retained_provider;

fn power(base: Value, exponent: Value) -> runmat_runtime::execution::RootedValue {
    block_on(runmat_runtime::call_builtin_async_with_outputs(
        "power",
        &[base, exponent],
        1,
    ))
    .unwrap()
}

#[test]
fn tiny_and_underflowed_imaginary_parts_keep_complex_storage() {
    for exponent in [0.5, 1.5] {
        let output = power(
            Value::Tensor(Tensor::new(vec![-1e-300, 4.0], vec![1, 2]).unwrap()),
            Value::Num(exponent),
        );
        let Value::ComplexTensor(output) = output.value() else {
            panic!("complex double array required")
        };
        let values = output.as_f64_slice().unwrap();
        if exponent == 0.5 {
            assert!(values[0].1 > 0.0);
            assert!((values[0].1 / 1e-150 - 1.0).abs() < 1e-12);
        }
        assert_eq!(values[1].1, 0.0);
    }
    for mixed in [false, true] {
        let base = Value::Tensor(Tensor::from_f32(vec![-1e-30, 4.0], vec![1, 2]).unwrap());
        let exponent = if mixed {
            Value::Tensor(Tensor::new(vec![0.5, 0.5], vec![1, 2]).unwrap())
        } else {
            Value::Tensor(Tensor::from_f32(vec![0.5, 0.5], vec![1, 2]).unwrap())
        };
        let result = power(base, exponent);
        let Value::ComplexTensor(output) = result.value() else {
            panic!("complex single required")
        };
        let values = output.as_f32_slice().unwrap();
        assert!(values[0].1 > 0.0);
        assert!((values[0].1 / 1e-15 - 1.0).abs() < 1e-5);
    }
}

#[test]
fn nan_propagation_is_identical_for_scalar_and_tensor_paths() {
    for (base, exponent) in [(1.0, f64::NAN), (f64::NAN, 0.0), (-1.0, f64::NAN)] {
        let result = power(Value::Num(base), Value::Num(exponent));
        let Value::Num(value) = result.value() else {
            panic!("real NaN")
        };
        assert!(value.is_nan());
        let result = power(
            Value::Tensor(Tensor::new(vec![base], vec![1, 1]).unwrap()),
            Value::Num(exponent),
        );
        let Value::Num(value) = result.value() else {
            panic!("real NaN")
        };
        assert!(value.is_nan());
        let result = power(
            Value::Tensor(Tensor::from_f32(vec![base as f32], vec![1, 1]).unwrap()),
            Value::Num(exponent),
        );
        let Value::Tensor(value) = result.value() else {
            panic!("real single NaN")
        };
        assert!(value.as_f32_slice().unwrap()[0].is_nan());
    }
}

#[test]
fn exceptional_real_pairs_keep_signed_zero_and_infinity_policy() {
    for (base, exponent, expected) in [
        (-0.0, 3.0, -0.0),
        (-0.0, -3.0, f64::NEG_INFINITY),
        (-0.0, 0.5, 0.0),
        (0.0, -0.5, f64::INFINITY),
        (f64::NEG_INFINITY, 0.5, f64::INFINITY),
        (f64::NEG_INFINITY, -3.0, -0.0),
        (-1.0, f64::INFINITY, 1.0),
    ] {
        let result = power(Value::Num(base), Value::Num(exponent));
        let Value::Num(actual) = result.value() else {
            panic!("real result")
        };
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
fn classification_uses_paired_broadcast_positions_not_unrelated_extrema() {
    let base = || Value::Tensor(Tensor::new(vec![-4.0, 4.0], vec![1, 2]).unwrap());
    let exponent = Value::Tensor(Tensor::new(vec![2.0, 0.5], vec![1, 2]).unwrap());
    let output = power(base(), exponent);
    let Value::Tensor(result) = output.value() else {
        panic!("all paired results real")
    };
    assert_eq!(result.as_f64_slice().unwrap(), &[16.0, 2.0]);
    let exponent = Value::Tensor(Tensor::new(vec![2.0, 0.5], vec![2, 1]).unwrap());
    let output = power(base(), exponent);
    let Value::ComplexTensor(result) = output.value() else {
        panic!("broadcast includes complex pair")
    };
    assert_eq!(result.shape(), &[2, 2]);
    assert!((result.as_f64_slice().unwrap()[1].1 - 2.0).abs() < 1e-12);
}

#[test]
fn actual_resident_dispatch_preserves_complex_promotion_in_both_compatibility_modes() {
    use runmat_accelerate::simple_provider::InProcessProvider;
    use runmat_accelerate_api::{
        AccelProvider, GpuTensorStorage, HostTensorView, ThreadProviderGuard,
    };
    let provider = retained_provider::retained_test_provider!(PROVIDER: InProcessProvider = InProcessProvider::new());
    let _provider = ThreadProviderGuard::set(Some(provider));
    for extensions in [false, true] {
        let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(extensions);
        let base = provider
            .upload(&HostTensorView {
                data: &[-1e-300, 4.0],
                shape: &[1, 2],
            })
            .unwrap();
        let exponent = provider
            .upload(&HostTensorView {
                data: &[0.5, 2.0],
                shape: &[1, 2],
            })
            .unwrap();
        for host_exponent in [false, true] {
            let exp = if host_exponent {
                Value::Num(0.5)
            } else {
                Value::GpuTensor(exponent.clone())
            };
            let result = power(Value::GpuTensor(base.clone()), exp);
            let Value::GpuTensor(output) = result.value() else {
                panic!("resident complex output")
            };
            assert_eq!(
                output.descriptor().storage,
                Some(GpuTensorStorage::ComplexInterleaved)
            );
            assert_eq!(output.shape(), &[1, 2]);
            let values = block_on(provider.download(output)).unwrap().data;
            assert_eq!(values.len(), 4);
            assert!((values[1] / 1e-150 - 1.0).abs() < 1e-12);
            assert_eq!(values[3], 0.0);
            result.with_owned(|value| {
                let Value::GpuTensor(output) = value else {
                    unreachable!("resident output was checked above")
                };
                output.release().unwrap();
            });
            assert_eq!(
                block_on(provider.download(&base)).unwrap().data,
                vec![-1e-300, 4.0]
            );
            assert_eq!(
                block_on(provider.download(&exponent)).unwrap().data,
                vec![0.5, 2.0]
            );
        }
        base.release().unwrap();
        exponent.release().unwrap();
    }
}
