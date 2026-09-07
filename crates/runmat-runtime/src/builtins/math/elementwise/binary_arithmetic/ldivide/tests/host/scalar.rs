use super::super::*;
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ldivide_scalar_numbers() {
    let result = ldivide_builtin(Value::Num(7.0), Value::Num(2.0), Vec::new()).expect("ldivide");
    match result {
        Value::Num(v) => assert!((v - (2.0 / 7.0)).abs() < EPS),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ldivide_integer_arrays_preserve_exact_reversed_division() {
    let divisor = Tensor::new_integer(IntegerStorage::I64(vec![2, 2, 0]), vec![1, 3])
        .expect("integer tensor");
    let numerator = Tensor::new_integer(IntegerStorage::I64(vec![-3, 3, i64::MAX]), vec![1, 3])
        .expect("integer tensor");
    let result = ldivide_builtin(Value::Tensor(divisor), Value::Tensor(numerator), Vec::new())
        .expect("integer ldivide");
    let Value::Tensor(result) = result else {
        panic!("expected typed integer tensor");
    };
    assert_eq!(
        result.integer_storage(),
        Some(&IntegerStorage::I64(vec![-2, 2, i64::MAX]))
    );
}

#[test]
fn ldivide_mixed_floating_arrays_return_single() {
    let divisor = Tensor::from_f32(vec![2.0, 4.0], vec![1, 2]).unwrap();
    let numerator = Tensor::new(vec![10.0, 20.0], vec![1, 2]).unwrap();

    let result =
        ldivide_builtin(Value::Tensor(divisor), Value::Tensor(numerator), Vec::new()).unwrap();
    let Value::Tensor(result) = result else {
        panic!("expected single tensor");
    };
    assert_eq!(
        result.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![5.0, 5.0])
    );
}

#[test]
fn ldivide_like_complex_conversion_reads_typed_integer_storage_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let tensor = Tensor::new_integer(IntegerStorage::U32(vec![4, 5]), vec![1, 2]).unwrap();

    let result = block_on(real_to_complex(
        OUTPUT_PROTOTYPE_CONTEXT,
        Value::Tensor(tensor),
    ))
    .expect("complex conversion");

    match result {
        Value::ComplexTensor(out) => {
            assert_eq!(out.shape, vec![1, 2]);
            assert_eq!(out.materialize_f64(), vec![(4.0, 0.0), (5.0, 0.0)]);
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[test]
fn ldivide_like_complex_conversion_preserves_single_storage() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let tensor = Tensor::from_f32(vec![4.0, 5.0], vec![1, 2]).unwrap();

    let result = block_on(real_to_complex(
        OUTPUT_PROTOTYPE_CONTEXT,
        Value::Tensor(tensor),
    ))
    .expect("complex conversion");

    let Value::ComplexTensor(result) = result else {
        panic!("expected complex single tensor");
    };
    assert_eq!(
        result.as_f32_slice().map(|values| values
            .iter()
            .copied()
            .map(<(f32, f32)>::from)
            .collect::<Vec<_>>()),
        Some(vec![(4.0, 0.0), (5.0, 0.0)])
    );
}

#[test]
fn ldivide_symbolic_scalar_returns_symbolic_quotient() {
    let result = ldivide_builtin(
        Value::Num(2.0),
        Value::Symbolic(SymbolicExpr::variable("x")),
        Vec::new(),
    )
    .expect("ldivide");

    assert_eq!(result.to_string(), "x/2");
}
