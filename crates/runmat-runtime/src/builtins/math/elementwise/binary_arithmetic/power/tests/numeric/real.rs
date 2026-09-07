use super::super::*;
use runmat_value::{CharArray, NumericStorage};
#[test]
fn power_float_arrays_preserve_native_single_class() {
    let base = Tensor::from_f32(vec![2.0, 4.0], vec![1, 2]).unwrap();
    let exponent = Tensor::new(vec![3.0, 0.5], vec![1, 2]).unwrap();
    let Value::Tensor(result) =
        power_builtin(Value::Tensor(base), Value::Tensor(exponent), Vec::new()).unwrap()
    else {
        panic!("expected single tensor");
    };
    assert_eq!(
        result.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![8.0, 2.0])
    );
}

#[test]
fn power_single_negative_base_promotes_to_complex_single_without_scalar_collapse() {
    let base = Tensor::from_f32(vec![-4.0], vec![1, 1]).unwrap();
    let exponent = Tensor::from_f32(vec![0.5], vec![1, 1]).unwrap();
    let result = power_builtin(Value::Tensor(base), Value::Tensor(exponent), Vec::new()).unwrap();
    let Value::ComplexTensor(result) = result else {
        panic!("expected one-element complex single tensor");
    };
    let values = result.as_f32_slice().expect("complex single");
    assert!(values[0].0.abs() < 1e-5);
    assert!((values[0].1 - 2.0).abs() < 1e-5);
}

#[test]
fn power_like_complex_conversion_reads_typed_integer_storage_exactly() {
    let tensor = Tensor::new_integer(IntegerStorage::I64(vec![-4, 5]), vec![1, 2]).unwrap();

    let result = block_on(real_to_complex(
        OUTPUT_PROTOTYPE_CONTEXT,
        Value::Tensor(tensor),
    ))
    .expect("complex conversion");

    match result {
        Value::ComplexTensor(out) => {
            assert_eq!(out.shape, vec![1, 2]);
            assert_eq!(out.materialize_f64(), vec![(-4.0, 0.0), (5.0, 0.0)]);
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn power_matrix_broadcast() {
    let base = Tensor::new((1..=3).map(|v| v as f64).collect::<Vec<_>>(), vec![3, 1]).unwrap();
    let exp = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let result = power_builtin(Value::Tensor(base), Value::Tensor(exp), Vec::new()).expect("power");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![3, 3]);
            let expected = [1.0, 2.0, 3.0, 1.0, 4.0, 9.0, 1.0, 8.0, 27.0];
            for (got, exp) in t
                .as_f64_slice()
                .expect("double result")
                .iter()
                .zip(expected.iter())
            {
                assert!((got - exp).abs() < 1e-12);
            }
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn power_char_array() {
    let chars = CharArray::new("AZ".chars().collect(), 1, 2).unwrap();
    let result =
        power_builtin(Value::CharArray(chars), Value::Num(2.0), Vec::new()).expect("power");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![1, 2]);
            let expected = [4225.0, 8100.0];
            for (got, exp) in t
                .as_f64_slice()
                .expect("double result")
                .iter()
                .zip(expected.iter())
            {
                assert!((got - exp).abs() < 1e-9);
            }
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn power_zero_negative_exponent_infinite() {
    let result = power_builtin(Value::Num(0.0), Value::Num(-2.0), Vec::new()).expect("power");
    match result {
        Value::Num(v) => assert!(v.is_infinite() && v.is_sign_positive()),
        other => panic!("expected scalar infinity, got {other:?}"),
    }
}
