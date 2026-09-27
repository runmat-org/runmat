use futures::executor::block_on;
use runmat_value::{CharArray, ComplexStorage, ComplexTensor, NumericDType, Tensor, Value};

fn call(name: &str, value: Value) -> Value {
    block_on(runmat_runtime::call_builtin_async_with_outputs(
        name,
        &[value],
        1,
    ))
    .unwrap()
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
        let Value::Num(value) = call(name, Value::Num(1.0 + 1e-14)) else {
            panic!("real result");
        };
        assert_eq!(value, expected);
        assert!(value > 0.0);
    }
    let Value::Num(value) = call("log1p", Value::Num(1e-300)) else {
        panic!("real result");
    };
    assert_eq!(value, 1e-300_f64.ln_1p());
    for name in ["log", "log2", "log10", "log1p"] {
        let scalar = call(name, Value::Complex(1.0, 1e-20));
        let tensor = call(
            name,
            Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 1e-20)], vec![1, 1]).unwrap()),
        );
        assert_eq!(scalar, tensor, "{name}");
        let Value::Complex(_, imaginary) = tensor else {
            panic!("complex result");
        };
        assert!(imaginary > 0.0, "{name}");
        let Value::Num(value) = call(name, Value::Num(f64::NAN)) else {
            panic!("real NaN");
        };
        assert!(value.is_nan());
    }
}

#[test]
fn square_root_preserves_small_results_and_nan_real_domain() {
    let Value::Num(value) = call("sqrt", Value::Num(1e-300)) else {
        panic!("real double");
    };
    assert_eq!(value, 1e-150);
    let Value::Complex(real, imag) = call("sqrt", Value::Num(-1e-300)) else {
        panic!("complex double");
    };
    assert_eq!(real, 0.0);
    assert_eq!(imag, 1e-150);
    let Value::Tensor(single) = call(
        "sqrt",
        Value::Tensor(Tensor::from_f32(vec![1e-30], vec![1, 1]).unwrap()),
    ) else {
        panic!("dense single");
    };
    assert!((single.as_f32_slice().unwrap()[0] / 1e-15 - 1.0).abs() < 1e-6);
    let Value::Num(value) = call("sqrt", Value::Num(f64::NAN)) else {
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
            matches!(real, Value::Tensor(ref value) if value.numeric_dtype() == NumericDType::F32),
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
                matches!(complex, Value::ComplexTensor(ref value) if value.numeric_dtype() == NumericDType::F32),
                "{name}"
            );
        }
    }
}

#[test]
fn character_pages_keep_shape_and_column_major_positions() {
    let _mode = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let chars =
        CharArray::new_with_shape(vec!['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], vec![2, 2, 2])
            .unwrap();
    for name in ["log", "log10", "log1p", "log2", "sqrt"] {
        let Value::Tensor(output) = call(name, Value::CharArray(chars.clone())) else {
            panic!("{name}: dense character result");
        };
        assert_eq!(output.shape, chars.shape, "{name}");
        for (actual, character) in output
            .as_f64_slice()
            .unwrap()
            .iter()
            .zip(chars.to_column_major())
        {
            let Value::Num(expected) = call(name, Value::Num(character as u32 as f64)) else {
                panic!("scalar result");
            };
            assert_eq!(*actual, expected, "{name}");
        }
    }
    let Value::OutputList(outputs) = block_on(runmat_runtime::call_builtin_async_with_outputs(
        "log2",
        &[Value::CharArray(chars.clone())],
        2,
    ))
    .unwrap() else {
        panic!("two outputs");
    };
    let [Value::Tensor(fractions), Value::Tensor(exponents)] = outputs.as_slice() else {
        panic!("dense dissection outputs");
    };
    assert_eq!(fractions.shape, chars.shape);
    assert_eq!(exponents.shape, chars.shape);
    for ((fraction, exponent), character) in fractions
        .as_f64_slice()
        .unwrap()
        .iter()
        .zip(exponents.as_f64_slice().unwrap())
        .zip(chars.to_column_major())
    {
        assert_eq!(fraction * 2.0_f64.powf(*exponent), character as u32 as f64);
    }
}
