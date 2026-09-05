use super::super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_string_input_has_stable_identifier() {
    let err = complex_call(Value::from("bad"), vec![]).expect_err("expected error");
    assert_eq!(err.identifier(), COMPLEX_ERROR_INVALID_INPUT.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_row_vector_pair() {
    let lhs = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let rhs = Tensor::new(vec![4.0, 5.0, 6.0], vec![1, 3]).unwrap();
    let result = complex_call(Value::Tensor(lhs), vec![Value::Tensor(rhs)]).expect("complex");
    match result {
        Value::ComplexTensor(ct) => {
            assert_eq!(ct.shape, vec![1, 3]);
            assert_eq!(
                ct.materialize_f64(),
                vec![(1.0, 4.0), (2.0, 5.0), (3.0, 6.0)]
            );
        }
        other => panic!("expected ComplexTensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_scalar_vector_broadcast_real_left() {
    let imag = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let result = complex_call(Value::Num(0.0), vec![Value::Tensor(imag)]).expect("complex");
    match result {
        Value::ComplexTensor(ct) => {
            assert_eq!(ct.shape, vec![1, 3]);
            assert_eq!(
                ct.materialize_f64(),
                vec![(0.0, 1.0), (0.0, 2.0), (0.0, 3.0)]
            );
        }
        other => panic!("expected ComplexTensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_scalar_vector_broadcast_real_right() {
    let real = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let result = complex_call(Value::Tensor(real), vec![Value::Num(0.0)]).expect("complex");
    match result {
        Value::ComplexTensor(ct) => {
            assert_eq!(ct.shape, vec![1, 3]);
            assert_eq!(
                ct.materialize_f64(),
                vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
            );
        }
        other => panic!("expected ComplexTensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_column_vectors() {
    let lhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let rhs = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
    let result = complex_call(Value::Tensor(lhs), vec![Value::Tensor(rhs)]).expect("complex");
    match result {
        Value::ComplexTensor(ct) => {
            assert_eq!(ct.shape, vec![2, 1]);
            assert_eq!(ct.materialize_f64(), vec![(1.0, 3.0), (2.0, 4.0)]);
        }
        other => panic!("expected ComplexTensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_rejects_non_scalar_implicit_expansion() {
    let row = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let col = Tensor::new(vec![10.0, 20.0], vec![2, 1]).unwrap();
    let err = complex_call(Value::Tensor(row), vec![Value::Tensor(col)]).unwrap_err();
    let msg = err.message().to_ascii_lowercase();
    assert!(msg.contains("same size") || msg.contains("scalar"), "{msg}");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_shape_mismatch_errors() {
    let lhs = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let rhs = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let err = complex_call(Value::Tensor(lhs), vec![Value::Tensor(rhs)]).unwrap_err();
    let msg = err.message().to_ascii_lowercase();
    assert!(msg.contains("dimension") || msg.contains("size"), "{msg}");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_rejects_complex_scalar() {
    let err = complex_call(Value::Complex(1.0, 2.0), vec![Value::Num(3.0)]).unwrap_err();
    assert!(
        err.message().contains("must be real"),
        "unexpected error: {}",
        err.message()
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_rejects_complex_imag_argument() {
    let err = complex_call(Value::Num(1.0), vec![Value::Complex(0.0, 1.0)]).unwrap_err();
    assert!(
        err.message().contains("must be real"),
        "unexpected error: {}",
        err.message()
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_rejects_complex_tensor_input() {
    let ct = ComplexTensor::new(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
    let err = complex_call(Value::ComplexTensor(ct), vec![Value::Num(0.0)]).unwrap_err();
    assert!(err.message().contains("must be real"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_rejects_string_input() {
    let err = complex_call(Value::from("hello"), vec![Value::Num(0.0)]).unwrap_err();
    assert!(err.message().contains("string"), "{}", err.message());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_rejects_string_array_input() {
    let arr = StringArray::new(vec!["a".to_string(), "b".to_string()], vec![1, 2]).expect("array");
    let err = complex_call(Value::Num(0.0), vec![Value::StringArray(arr)]).unwrap_err();
    assert!(err.message().contains("string"), "{}", err.message());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_unary_scalar_zero_imag() {
    let result = complex_call(Value::Num(5.0), Vec::new()).expect("complex");
    match result {
        Value::Complex(re, im) => {
            assert_eq!(re, 5.0);
            assert_eq!(im, 0.0);
        }
        other => panic!("expected Complex result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_unary_tensor_zero_imag() {
    let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let result = complex_call(Value::Tensor(tensor), Vec::new()).expect("complex");
    match result {
        Value::ComplexTensor(ct) => {
            assert_eq!(ct.shape, vec![1, 2]);
            assert_eq!(ct.materialize_f64(), vec![(1.0, 0.0), (2.0, 0.0)]);
        }
        other => panic!("expected ComplexTensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_unary_complex_scalar_passthrough() {
    let result = complex_call(Value::Complex(1.0, 2.0), Vec::new()).expect("complex");
    match result {
        Value::Complex(re, im) => {
            assert_eq!(re, 1.0);
            assert_eq!(im, 2.0);
        }
        other => panic!("expected Complex result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_unary_complex_tensor_passthrough() {
    let tensor = ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![1, 2]).unwrap();
    let result = complex_call(Value::ComplexTensor(tensor.clone()), Vec::new()).expect("complex");
    match result {
        Value::ComplexTensor(out) => assert_eq!(out, tensor),
        other => panic!("expected ComplexTensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_unary_rejects_string_input() {
    let err = complex_call(Value::from("hi"), Vec::new()).unwrap_err();
    assert!(err.message().contains("string"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_logical_array_input() {
    let lhs = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
    let rhs = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]).unwrap();
    let result = complex_call(Value::LogicalArray(lhs), vec![Value::Tensor(rhs)]).expect("complex");
    match result {
        Value::ComplexTensor(ct) => {
            assert_eq!(ct.shape, vec![2, 2]);
            assert_eq!(
                ct.materialize_f64(),
                vec![(1.0, 10.0), (0.0, 20.0), (0.0, 30.0), (1.0, 40.0)]
            );
        }
        other => panic!("expected ComplexTensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_bool_scalar_promotion() {
    let result = complex_call(Value::Bool(true), vec![Value::Bool(false)]).expect("complex");
    match result {
        Value::Complex(re, im) => {
            assert_eq!(re, 1.0);
            assert_eq!(im, 0.0);
        }
        other => panic!("expected Complex result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_rejects_char_array_input() {
    let chars = CharArray::new("AB".chars().collect(), 1, 2).unwrap();
    let imag = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let err = complex_call(Value::CharArray(chars), vec![Value::Tensor(imag)]).unwrap_err();
    assert!(err.message().contains("char"), "{}", err.message());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_empty_tensor_inputs() {
    let lhs = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
    let rhs = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
    let result = complex_call(Value::Tensor(lhs), vec![Value::Tensor(rhs)]).expect("complex");
    match result {
        Value::ComplexTensor(ct) => {
            assert_eq!(ct.shape, vec![0, 3]);
            assert!(ct.materialize_f64().is_empty());
        }
        other => panic!("expected empty ComplexTensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_too_many_args_errors() {
    let err = complex_call(Value::Num(1.0), vec![Value::Num(2.0), Value::Num(3.0)]).unwrap_err();
    assert!(
        err.message().contains("1 or 2 input arguments"),
        "{}",
        err.message()
    );
}
