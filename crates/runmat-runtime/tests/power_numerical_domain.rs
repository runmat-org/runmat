use futures::executor::block_on;
use runmat_value::{Tensor, Value};

fn power(base: Value, exponent: Value) -> Value {
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
        let Value::ComplexTensor(output) = output else {
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
        let Value::ComplexTensor(output) = power(base, exponent) else {
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
        let Value::Num(value) = power(Value::Num(base), Value::Num(exponent)) else {
            panic!("real NaN")
        };
        assert!(value.is_nan());
        let Value::Num(value) = power(
            Value::Tensor(Tensor::new(vec![base], vec![1, 1]).unwrap()),
            Value::Num(exponent),
        ) else {
            panic!("real NaN")
        };
        assert!(value.is_nan());
        let Value::Tensor(value) = power(
            Value::Tensor(Tensor::from_f32(vec![base as f32], vec![1, 1]).unwrap()),
            Value::Num(exponent),
        ) else {
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
        let Value::Num(actual) = power(Value::Num(base), Value::Num(exponent)) else {
            panic!("real result")
        };
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
fn classification_uses_paired_broadcast_positions_not_unrelated_extrema() {
    let base = || Value::Tensor(Tensor::new(vec![-4.0, 4.0], vec![1, 2]).unwrap());
    let exponent = Value::Tensor(Tensor::new(vec![2.0, 0.5], vec![1, 2]).unwrap());
    let Value::Tensor(result) = power(base(), exponent) else {
        panic!("all paired results real")
    };
    assert_eq!(result.as_f64_slice().unwrap(), &[16.0, 2.0]);
    let exponent = Value::Tensor(Tensor::new(vec![2.0, 0.5], vec![2, 1]).unwrap());
    let Value::ComplexTensor(result) = power(base(), exponent) else {
        panic!("broadcast includes complex pair")
    };
    assert_eq!(result.shape, vec![2, 2]);
    assert!((result.as_f64_slice().unwrap()[1].1 - 2.0).abs() < 1e-12);
}

#[test]
fn actual_resident_dispatch_preserves_complex_promotion_in_both_compatibility_modes() {
    use runmat_accelerate::simple_provider::InProcessProvider;
    use runmat_accelerate_api::{
        AccelProvider, GpuTensorStorage, HostTensorView, ThreadProviderGuard,
    };
    use std::sync::OnceLock;
    static PROVIDER: OnceLock<InProcessProvider> = OnceLock::new();
    let provider = PROVIDER.get_or_init(InProcessProvider::new);
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
            let Value::GpuTensor(output) = power(Value::GpuTensor(base.clone()), exp) else {
                panic!("resident complex output")
            };
            assert_eq!(
                output.descriptor.storage,
                Some(GpuTensorStorage::ComplexInterleaved)
            );
            let values = block_on(provider.download(&output)).unwrap().data;
            assert!((values[1] / 1e-150 - 1.0).abs() < 1e-12);
            assert_eq!(values[3], 0.0);
            provider.free(&output).unwrap();
        }
        provider.free(&base).unwrap();
        provider.free(&exponent).unwrap();
    }
}
