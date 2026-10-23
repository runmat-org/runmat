use futures::executor::block_on;
use runmat_value::{CharArray, ComplexStorage, ComplexTensor, NumericDType, Tensor, Value};

fn call(name: &str, value: Value) -> runmat_runtime::execution::RootedValue {
    block_on(runmat_runtime::call_builtin_async_with_outputs(
        name,
        &[value],
        1,
    ))
    .unwrap()
}

#[test]
fn complex_roots_retain_small_components_and_finite_extremes() {
    for sign in [-1.0, 1.0] {
        let result = call("sqrt", Value::Complex(1.0, sign * 1e-20));
        let Value::Complex(real, imag) = result.value() else {
            panic!("complex scalar result");
        };
        assert_eq!(*real, 1.0);
        assert!((imag / (sign * 5e-21) - 1.0).abs() < 4.0 * f64::EPSILON);
    }
    // Polar half-angle identities supply an oracle without forming |MAX+iMAX|.
    let scale = f64::MAX.sqrt();
    let result = call("sqrt", Value::Complex(f64::MAX, f64::MAX));
    let Value::Complex(real, imag) = result.value() else {
        panic!("complex scalar result");
    };
    let radius = 2.0_f64.powf(0.25);
    let angle = std::f64::consts::FRAC_PI_8;
    assert!((real / scale - radius * angle.cos()).abs() < 1e-15);
    assert!((imag / scale - radius * angle.sin()).abs() < 1e-15);
    let result = call("sqrt", Value::Complex(f64::MAX, 1.0));
    let Value::Complex(real, imag) = result.value() else {
        panic!("complex scalar result");
    };
    assert!((real / scale - 1.0).abs() < 4.0 * f64::EPSILON);
    assert!((imag * scale - 0.5).abs() < 4.0 * f64::EPSILON);

    let input = ComplexTensor::from_complex_storage(
        ComplexStorage::F32(vec![(1.0, 1e-20), (f32::MAX, f32::MAX)].into()),
        vec![1, 1, 2],
    )
    .unwrap();
    let result = call("sqrt", Value::ComplexTensor(input));
    let Value::ComplexTensor(output) = result.value() else {
        panic!("single complex pages");
    };
    assert_eq!(output.shape(), &[1, 1, 2]);
    let ComplexStorage::F32(values) = output.complex_storage() else {
        panic!("single retained")
    };
    assert!((values[0].1 / 5e-21 - 1.0).abs() < 4.0 * f32::EPSILON);
    assert!(values[1].0.is_finite() && values[1].1.is_finite());
}

#[test]
fn complex_logarithms_do_not_overflow_finite_magnitudes() {
    for (name, multiplier) in [
        ("log", 1.0),
        ("log1p", 1.0),
        ("log2", std::f64::consts::LOG2_E),
        ("log10", std::f64::consts::LOG10_E),
    ] {
        let result = call(name, Value::Complex(f64::MAX, f64::MAX));
        let Value::Complex(real, imag) = result.value() else {
            panic!("{name}: complex result");
        };
        let expected = (f64::MAX.ln() + std::f64::consts::LN_2 * 0.5) * multiplier;
        assert!(
            (real - expected).abs() < 4.0 * f64::EPSILON * expected,
            "{name}"
        );
        assert!(
            (imag - std::f64::consts::FRAC_PI_4 * multiplier).abs() < 1e-15,
            "{name}"
        );
        let input = ComplexTensor::from_complex_storage(
            ComplexStorage::F32(vec![(f32::MAX, f32::MAX)].into()),
            vec![1, 1],
        )
        .unwrap();
        let result = call(name, Value::ComplexTensor(input));
        let Value::ComplexTensor(output) = result.value() else {
            panic!("single complex")
        };
        let ComplexStorage::F32(values) = output.complex_storage() else {
            panic!("single retained")
        };
        let expected = (f64::from(f32::MAX).ln() + std::f64::consts::LN_2 * 0.5) * multiplier;
        assert!(
            (f64::from(values[0].0) - expected).abs() < 4.0 * f64::from(f32::EPSILON) * expected
        );
    }
}

#[test]
fn discarded_outputs_still_validate_argument_and_domain() {
    for name in ["log", "log10", "log1p", "log2", "sqrt", "realsqrt"] {
        assert!(
            block_on(runmat_runtime::call_builtin_async_with_outputs(
                name,
                &[Value::String("not numeric".into())],
                0,
            ))
            .is_err(),
            "{name}"
        );
        assert!(
            block_on(runmat_runtime::call_builtin_async_with_outputs(
                name,
                &[Value::Num(4.0)],
                0,
            ))
            .is_ok(),
            "{name}"
        );
    }
    assert!(block_on(runmat_runtime::call_builtin_async_with_outputs(
        "realsqrt",
        &[Value::Num(-1.0)],
        0,
    ))
    .is_err());
}

#[test]
fn logarithms_retain_small_real_and_complex_components() {
    for (name, expected) in [
        ("log", (1.0_f64 + 1e-14).ln()),
        ("log2", (1.0_f64 + 1e-14).log2()),
        ("log10", (1.0_f64 + 1e-14).log10()),
    ] {
        let result = call(name, Value::Num(1.0 + 1e-14));
        let Value::Num(value) = result.value() else {
            panic!("real result");
        };
        assert_eq!(*value, expected);
        assert!(*value > 0.0);
    }
    let result = call("log1p", Value::Num(1e-300));
    let Value::Num(value) = result.value() else {
        panic!("real result");
    };
    assert_eq!(*value, 1e-300_f64.ln_1p());
    for name in ["log", "log2", "log10", "log1p"] {
        let scalar = call(name, Value::Complex(1.0, 1e-20));
        let tensor = call(
            name,
            Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 1e-20)], vec![1, 1]).unwrap()),
        );
        assert_eq!(scalar, tensor, "{name}");
        let Value::Complex(_, imaginary) = tensor.value() else {
            panic!("complex result");
        };
        assert!(*imaginary > 0.0, "{name}");
        let result = call(name, Value::Num(f64::NAN));
        let Value::Num(value) = result.value() else {
            panic!("real NaN");
        };
        assert!(value.is_nan());
    }
}

#[test]
fn square_root_preserves_small_results_and_nan_real_domain() {
    let result = call("sqrt", Value::Num(1e-300));
    let Value::Num(value) = result.value() else {
        panic!("real double");
    };
    assert_eq!(*value, 1e-150);
    let result = call("sqrt", Value::Num(-1e-300));
    let Value::Complex(real, imag) = result.value() else {
        panic!("complex double");
    };
    assert_eq!(*real, 0.0);
    assert_eq!(*imag, 1e-150);
    let result = call(
        "sqrt",
        Value::Tensor(Tensor::from_f32(vec![1e-30], vec![1, 1]).unwrap()),
    );
    let Value::Tensor(single) = result.value() else {
        panic!("dense single");
    };
    assert!((single.as_f32_slice().unwrap()[0] / 1e-15 - 1.0).abs() < 1e-6);
    let result = call("sqrt", Value::Num(f64::NAN));
    let Value::Num(value) = result.value() else {
        panic!("real NaN");
    };
    assert!(value.is_nan());
}

#[test]
fn floating_singleton_runtime_carriers_match_inference() {
    for name in ["log", "log10", "log1p", "log2", "sqrt", "realsqrt"] {
        let real = call(
            name,
            Value::Tensor(Tensor::from_f32(vec![4.0], vec![1, 1]).unwrap()),
        );
        assert!(
            matches!(real.value(), Value::Tensor(value) if value.numeric_dtype() == NumericDType::F32),
            "{name}"
        );
        if name != "realsqrt" {
            let complex = call(
                name,
                Value::ComplexTensor(
                    ComplexTensor::from_complex_storage(
                        ComplexStorage::F32(vec![(4.0, 1.0)].into()),
                        vec![1, 1],
                    )
                    .unwrap(),
                ),
            );
            assert!(
                matches!(complex.value(), Value::ComplexTensor(value) if value.numeric_dtype() == NumericDType::F32),
                "{name}"
            );
        }
    }
}

#[test]
fn character_pages_keep_shape_and_column_major_positions() {
    let _mode = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let chars = CharArray::from_unicode_scalars_with_shape(
        vec!['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
        vec![2, 2, 2],
    )
    .unwrap();
    for name in ["log", "log10", "log1p", "log2", "sqrt"] {
        let result = call(name, Value::CharArray(chars.clone()));
        let Value::Tensor(output) = result.value() else {
            panic!("{name}: dense character result");
        };
        assert_eq!(output.shape(), &chars.shape[..], "{name}");
        for (actual, character) in output
            .as_f64_slice()
            .unwrap()
            .iter()
            .zip(chars.to_column_major())
        {
            let scalar = call(name, Value::Num(f64::from(character.raw())));
            let Value::Num(expected) = scalar.value() else {
                panic!("scalar result");
            };
            assert_eq!(*actual, *expected, "{name}");
        }
    }
    let result = block_on(runmat_runtime::call_builtin_async_with_outputs(
        "log2",
        &[Value::CharArray(chars.clone())],
        2,
    ))
    .unwrap();
    let Value::OutputList(outputs) = result.value() else {
        panic!("two outputs");
    };
    let [Value::Tensor(fractions), Value::Tensor(exponents)] = outputs.as_slice() else {
        panic!("dense dissection outputs");
    };
    assert_eq!(fractions.shape(), &chars.shape[..]);
    assert_eq!(exponents.shape(), &chars.shape[..]);
    for ((fraction, exponent), character) in fractions
        .as_f64_slice()
        .unwrap()
        .iter()
        .zip(exponents.as_f64_slice().unwrap())
        .zip(chars.to_column_major())
    {
        assert_eq!(
            fraction * 2.0_f64.powf(*exponent),
            f64::from(character.raw())
        );
    }
}
