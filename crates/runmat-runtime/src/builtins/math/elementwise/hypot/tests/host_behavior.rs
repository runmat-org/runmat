use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn hypot_scalar_pair() {
    let result = hypot_builtin(Value::Num(3.0), Value::Num(4.0)).expect("hypot");
    match result {
        Value::Num(v) => assert!((v - 5.0).abs() < 1e-12),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn hypot_matrix_elements() {
    let lhs = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
    let rhs = Tensor::new(vec![0.0, 0.0, 1.0, 1.0], vec![2, 2]).unwrap();
    let result = hypot_builtin(Value::Tensor(lhs), Value::Tensor(rhs)).expect("element-wise hypot");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            let expected = [1.0, 3.0, (5.0f64).sqrt(), (17.0f64).sqrt()];
            for (actual, expect) in t.materialize_f64().iter().zip(expected.iter()) {
                assert!((actual - expect).abs() < 1e-12, "{actual} vs {expect}");
            }
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn hypot_scalar_broadcast() {
    let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let result = hypot_builtin(Value::Tensor(matrix), Value::Num(4.0)).expect("broadcast");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            let expected = [4.123105625617661, 4.47213595499958, 5.0, 5.656854249492381];
            for (actual, expect) in t.materialize_f64().iter().zip(expected.iter()) {
                assert!((actual - expect).abs() < 1e-12);
            }
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn hypot_row_vector_broadcasts_over_matrix() {
    let matrix = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
    let row = Tensor::new(vec![3.0, 4.0, 5.0], vec![1, 3]).unwrap();
    let result = hypot_builtin(Value::Tensor(matrix), Value::Tensor(row)).expect("row broadcast");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 3]);
            let expected = [
                (1.0f64).hypot(3.0),
                (4.0f64).hypot(3.0),
                (2.0f64).hypot(4.0),
                (5.0f64).hypot(4.0),
                (3.0f64).hypot(5.0),
                (6.0f64).hypot(5.0),
            ];
            for (actual, expect) in t.materialize_f64().iter().zip(expected.iter()) {
                assert!((actual - expect).abs() < 1e-12, "{actual} vs {expect}");
            }
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn hypot_nan_propagates() {
    let result = hypot_builtin(Value::Num(f64::NAN), Value::Num(1.0)).expect("nan propagation");
    match result {
        Value::Num(v) => assert!(v.is_nan()),
        other => panic!("expected NaN scalar, got {other:?}"),
    }
}

#[test]
fn hypot_nan_takes_precedence_over_infinity() {
    for (lhs, rhs) in [
        (f64::NAN, f64::INFINITY),
        (f64::INFINITY, f64::NAN),
        (f64::NAN, f64::NEG_INFINITY),
        (f64::NEG_INFINITY, f64::NAN),
    ] {
        let result = hypot_builtin(Value::Num(lhs), Value::Num(rhs)).unwrap();
        assert!(matches!(result, Value::Num(value) if value.is_nan()));
    }
}

/// IEEE 754 / MATLAB require hypot(Inf, Inf) = Inf.
/// The scaling form lo/hi = Inf/Inf = NaN, so the host path must not regress.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn hypot_both_infinite_is_inf() {
    let result =
        hypot_builtin(Value::Num(f64::INFINITY), Value::Num(f64::INFINITY)).expect("hypot inf");
    match result {
        Value::Num(v) => assert!(v.is_infinite() && v > 0.0, "expected +Inf, got {v}"),
        other => panic!("expected +Inf scalar, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn hypot_one_infinite_is_inf() {
    let result =
        hypot_builtin(Value::Num(f64::INFINITY), Value::Num(3.0)).expect("hypot inf/finite");
    match result {
        Value::Num(v) => assert!(v.is_infinite() && v > 0.0, "expected +Inf, got {v}"),
        other => panic!("expected +Inf scalar, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn hypot_empty_tensor_result() {
    let lhs = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
    let rhs = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
    let result = hypot_builtin(Value::Tensor(lhs), Value::Tensor(rhs)).expect("empty hypot result");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape, vec![0, 3]);
            assert!(out.materialize_f64().is_empty());
        }
        other => panic!("expected empty tensor, got {other:?}"),
    }
}
