use super::super::*;
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ldivide_matrix_scalar() {
    let tensor = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2]).unwrap();
    let result =
        ldivide_builtin(Value::Tensor(tensor), Value::Num(2.0), Vec::new()).expect("ldivide");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            let expected = [1.0, 0.5, 0.3333333333333333, 0.25];
            for (got, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                assert!((got - exp).abs() < 1e-12);
            }
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ldivide_row_column_broadcast() {
    let column = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let row = Tensor::new(vec![10.0, 20.0, 40.0], vec![1, 3]).unwrap();
    let result = ldivide_builtin(Value::Tensor(column), Value::Tensor(row), Vec::new())
        .expect("broadcast ldivide");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![3, 3]);
            let expected = [
                10.0,
                5.0,
                3.3333333333333335,
                20.0,
                10.0,
                6.666666666666667,
                40.0,
                20.0,
                13.333333333333334,
            ];
            for (got, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                assert!((got - exp).abs() < EPS);
            }
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ldivide_division_by_zero() {
    let tensor = Tensor::new(vec![0.0, 1.0, -2.0], vec![3, 1]).unwrap();
    let result =
        ldivide_builtin(Value::Tensor(tensor), Value::Num(0.0), Vec::new()).expect("ldivide");
    match result {
        Value::Tensor(t) => {
            assert!(t.materialize_f64()[0].is_nan());
            assert_eq!(t.materialize_f64()[1], 0.0);
            assert_eq!(t.materialize_f64()[2], -0.0);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ldivide_logical_inputs_promote() {
    let logical = LogicalArray::new(vec![1, 0, 1, 1], vec![2, 2]).unwrap();
    let tensor = Tensor::new(vec![1.0, 2.0, 4.0, 8.0], vec![2, 2]).unwrap();
    let result = ldivide_builtin(
        Value::LogicalArray(logical),
        Value::Tensor(tensor),
        Vec::new(),
    )
    .expect("logical ldivide");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            let expected = [1.0, f64::INFINITY, 4.0, 8.0];
            for (got, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                if exp.is_infinite() {
                    assert!(got.is_infinite());
                } else {
                    assert!((got - exp).abs() < EPS);
                }
            }
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ldivide_char_array_promotes_to_double() {
    let chars = CharArray::new_row("AB");
    let result =
        ldivide_builtin(Value::CharArray(chars), Value::Num(2.0), Vec::new()).expect("ldivide");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![1, 2]);
            assert!((t.materialize_f64()[0] - (2.0 / 65.0)).abs() < EPS);
            assert!((t.materialize_f64()[1] - (2.0 / 66.0)).abs() < EPS);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ldivide_same_class_integer_inputs_preserve_class() {
    let lhs = Value::Int(IntValue::I32(6));
    let rhs = Value::Int(IntValue::I32(4));
    let result = ldivide_builtin(lhs, rhs, Vec::new()).expect("ldivide");
    assert_eq!(result, Value::Int(IntValue::I32(1)));
}
