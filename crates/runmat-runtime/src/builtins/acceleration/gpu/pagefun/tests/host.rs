use super::*;
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pagefun_mtimes_single_page() {
    let lhs = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
    let rhs = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();
    let result = pagefun_impl(
        Value::FunctionHandle("mtimes".into()),
        Value::Tensor(lhs),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(rhs)]),
    );
    let result = block_on(result).expect("pagefun");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2]);
            assert_eq!(t.materialize_f64(), vec![19.0, 43.0, 22.0, 50.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn pagefun_preserves_rank_zero_and_rank_one_canonical_matrix_shapes() {
    let scalar = block_on(pagefun_impl(
        Value::FunctionHandle("mtimes".into()),
        Value::Tensor(Tensor::new(vec![2.0], Vec::new()).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(3.0)]),
    ))
    .expect("rank-zero matrix product");
    let Value::Tensor(scalar) = scalar else {
        panic!("expected real tensor result");
    };
    assert_eq!(scalar.shape(), &[1, 1]);
    assert_eq!(scalar.materialize_f64(), vec![6.0]);

    let vector = block_on(pagefun_impl(
        Value::FunctionHandle("mtimes".into()),
        Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![2]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(2.0)]),
    ))
    .expect("rank-one complex matrix product");
    let Value::ComplexTensor(vector) = vector else {
        panic!("expected complex tensor result");
    };
    assert_eq!(vector.shape(), &[1, 2]);
    assert_eq!(vector.materialize_f64(), vec![(2.0, 4.0), (6.0, 8.0)]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pagefun_mtimes_multiple_pages() {
    let lhs = Tensor::new(vec![1.0, 3.0, 2.0, 4.0, 2.0, 1.0, 0.0, 3.0], vec![2, 2, 2]).unwrap();
    let rhs = Tensor::new(vec![5.0, 7.0, 6.0, 8.0, 1.0, 0.0, 2.0, 1.0], vec![2, 2, 2]).unwrap();
    let result = pagefun_impl(
        Value::from("@mtimes"),
        Value::Tensor(lhs),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(rhs)]),
    );
    let result = block_on(result).expect("pagefun");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2, 2]);
            assert_eq!(
                t.materialize_f64(),
                vec![19.0, 43.0, 22.0, 50.0, 2.0, 1.0, 4.0, 5.0]
            );
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pagefun_mtimes_broadcast_rhs() {
    let lhs = Tensor::new(vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0], vec![2, 2, 2]).unwrap();
    let rhs = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
    let result = pagefun_impl(
        Value::FunctionHandle("mtimes".into()),
        Value::Tensor(lhs),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(rhs)]),
    );
    let result = block_on(result).expect("pagefun");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2, 2]);
            assert_eq!(
                t.materialize_f64(),
                vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0],
                "broadcasted identity should preserve pages"
            );
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pagefun_mtimes_empty_pages() {
    let lhs = Tensor::new(Vec::new(), vec![2, 2, 0]).unwrap();
    let rhs = Tensor::new(Vec::new(), vec![2, 2, 0]).unwrap();
    let result = pagefun_impl(
        Value::from("@mtimes"),
        Value::Tensor(lhs),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(rhs)]),
    );
    let result = block_on(result).expect("pagefun");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2, 0]);
            assert!(t.materialize_f64().is_empty());
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pagefun_page_dimension_mismatch() {
    let lhs = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]).unwrap();
    let rhs = Tensor::new(
        vec![
            1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        vec![2, 2, 3],
    )
    .unwrap();
    let err = pagefun_impl(
        Value::FunctionHandle("mtimes".into()),
        Value::Tensor(lhs),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(rhs)]),
    );
    let err = block_on(err).expect_err("expected page dimension mismatch");
    assert!(
        err.contains("page dimension"),
        "unexpected mismatch error message: {err}"
    );
    assert_eq!(err.identifier(), PAGEFUN_ERROR_PAGE_DIM_MISMATCH.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pagefun_mtimes_dim_mismatch() {
    let lhs = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let rhs = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let err = pagefun_impl(
        Value::FunctionHandle("mtimes".into()),
        Value::Tensor(lhs),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(rhs)]),
    );
    let err = block_on(err).expect_err("expected dimension mismatch");
    assert!(
        err.contains("inner matrix dimensions"),
        "unexpected error message {err}"
    );
    assert_eq!(
        err.identifier(),
        PAGEFUN_ERROR_MATRIX_DIM_MISMATCH.identifier
    );
}

#[test]
fn pagefun_type_names_report_authoritative_numeric_storage() {
    let single = Value::Tensor(Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap());
    let integer = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap(),
    );
    let complex_single =
        Value::ComplexTensor(ComplexTensor::from_f32(vec![(1.0, -2.0)], vec![1, 1]).unwrap());

    assert_eq!(single.type_name(), "single array");
    assert_eq!(integer.type_name(), "uint64 array");
    assert_eq!(complex_single.type_name(), "complex single array");
}
