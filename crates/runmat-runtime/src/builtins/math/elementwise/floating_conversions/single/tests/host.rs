use super::*;

#[test]
fn single_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = SINGLE_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"Y = single(X)"));
    assert!(labels.contains(&"Y = single(X, \"like\", prototype)"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_scalar_rounds_to_f32() {
    let value = Value::Num(std::f64::consts::PI);
    let result = single_builtin(value, Vec::new()).expect("single");
    let Value::Tensor(tensor) = result else {
        panic!("expected native single scalar tensor");
    };
    assert_eq!(tensor.shape, vec![1, 1]);
    assert_eq!(
        tensor.as_f32_slice(),
        Some(&[std::f64::consts::PI as f32][..])
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_converts_symbolic_constants() {
    let result = single_builtin(
        Value::Symbolic(SymbolicExpr::constant(std::f64::consts::PI)),
        Vec::new(),
    )
    .expect("single");

    let Value::Tensor(tensor) = result else {
        panic!("expected native single symbolic constant");
    };
    assert_eq!(
        tensor.as_f32_slice(),
        Some(&[std::f64::consts::PI as f32][..])
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_converts_symbolic_array_constants() {
    let array = SymbolicArray::new(
        vec![SymbolicExpr::constant(1.25), SymbolicExpr::constant(2.5)],
        vec![1, 2],
    )
    .unwrap();

    let result = single_builtin(Value::SymbolicArray(array), Vec::new()).expect("single");

    match result {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape, vec![1, 2]);
            assert_eq!(
                tensor.materialize_f64(),
                vec![(1.25f32) as f64, (2.5f32) as f64]
            );
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_rejects_symbolic_array_variables() {
    let array = SymbolicArray::new(vec![SymbolicExpr::variable("x")], vec![1, 1]).unwrap();

    let err = single_builtin(Value::SymbolicArray(array), Vec::new())
        .expect_err("symbolic variable should not convert");

    assert_eq!(err.identifier(), SINGLE_ERROR_INVALID_INPUT.identifier);
    assert!(err.message().contains("conversion to single from sym"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_rejects_symbolic_variables() {
    let err = single_builtin(Value::Symbolic(SymbolicExpr::variable("x")), Vec::new())
        .expect_err("symbolic variable should not convert");

    assert_eq!(err.identifier(), SINGLE_ERROR_INVALID_INPUT.identifier);
    assert!(err.message().contains("conversion to single from sym"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_tensor_preserves_shape() {
    let tensor = Tensor::new(vec![1.25, 2.5, 3.75, 4.5], vec![2, 2]).unwrap();
    let result = single_builtin(Value::Tensor(tensor), Vec::new()).expect("single");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            assert_eq!(
                t.into_numeric_storage().expect("single storage"),
                NumericStorage::F32(vec![1.25, 2.5, 3.75, 4.5])
            );
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_tensor_reads_typed_integer_storage_exactly() {
    let tensor = Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 3, 4]), vec![2, 2]).unwrap();

    let result = single_builtin(Value::Tensor(tensor), Vec::new()).expect("single");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            assert_eq!(
                t.into_numeric_storage().expect("single storage"),
                NumericStorage::F32(vec![1.0, 2.0, 3.0, 4.0])
            );
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_sparse_tensor_preserves_sparsity_and_converts_storage() {
    let sparse = SparseTensor::new(
        3,
        2,
        vec![0, 1, 2],
        vec![1, 2],
        vec![std::f64::consts::PI, -1.25],
    )
    .unwrap();
    let result = single_builtin(Value::SparseTensor(sparse), Vec::new()).expect("single");
    let Value::SparseTensor(output) = result else {
        panic!("expected sparse tensor");
    };
    assert_eq!(output.shape(), vec![3, 2]);
    assert_eq!(
        output.numeric_dtype(),
        Some(runmat_value::NumericDType::F32)
    );
    assert_eq!(
        output.as_f32_slice(),
        Some(&[std::f64::consts::PI as f32, -1.25][..])
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_complex_tensor_rounds_both_components() {
    let tensor = ComplexTensor::new(
        vec![(1.234567, -9.876543), (0.3333333, 0.6666667)],
        vec![1, 2],
    )
    .unwrap();
    let result = single_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("single");
    match result {
        Value::ComplexTensor(t) => {
            assert_eq!(t.shape, vec![1, 2]);
            let expected: Vec<(f64, f64)> = vec![
                ((1.234567f32) as f64, (-9.876543f32) as f64),
                ((0.3333333f32) as f64, (0.6666667f32) as f64),
            ];
            assert_eq!(t.materialize_f64(), expected);
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_complex_tensor_reads_typed_integer_storage_exactly() {
    let storage = IntegerComplexStorage::new(
        IntegerStorage::I16(vec![1, -2]),
        IntegerStorage::I16(vec![3, -4]),
    )
    .expect("complex integer storage");
    let tensor = ComplexTensor::new_integer(storage, vec![1, 2]).unwrap();

    let result = single_builtin(Value::ComplexTensor(tensor), Vec::new()).expect("single");
    match result {
        Value::ComplexTensor(t) => {
            assert_eq!(t.shape, vec![1, 2]);
            assert_eq!(t.integer_storage(), None);
            assert_eq!(t.materialize_f64(), vec![(1.0, 3.0), (-2.0, -4.0)]);
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_char_array_produces_codes() {
    let chars = CharArray::new_row("AZ");
    let result = single_builtin(Value::CharArray(chars), Vec::new()).expect("single");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![1, 2]);
            assert_eq!(t.materialize_f64(), vec![65.0, 90.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn single_errors_on_string_input() {
    let err =
        single_builtin(Value::String("hello".to_string()), Vec::new()).expect_err("expected error");
    assert_eq!(err.identifier(), SINGLE_ERROR_INVALID_INPUT.identifier);
    assert!(err.message().contains("string"));
}
