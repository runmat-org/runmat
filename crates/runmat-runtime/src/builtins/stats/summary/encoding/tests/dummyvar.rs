use super::*;

#[test]
fn dummyvar_encodes_numeric_columns_and_missing() {
    let Value::Tensor(out) = block_on(dummyvar_builtin(tensor(
        vec![1.0, 2.0, f64::NAN, 1.0, 2.0, 2.0],
        vec![3, 2],
    )))
    .unwrap() else {
        panic!("tensor");
    };
    assert_eq!(out.shape(), &[3, 4]);
    assert_f64_slice_eq_nan(
        &out.materialize_f64(),
        &[
            1.0,
            0.0,
            f64::NAN,
            0.0,
            1.0,
            f64::NAN,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
        ],
    );
}

#[test]
fn dummyvar_encodes_cell_grouping_variables() {
    let phone = Value::Cell(
        CellArray::new(
            vec![
                Value::from("mobile"),
                Value::from("landline"),
                Value::from("mobile"),
            ],
            3,
            1,
        )
        .unwrap(),
    );
    let region = tensor(vec![2.0, 1.0, 2.0], vec![3, 1]);
    let Value::Tensor(out) = block_on(dummyvar_builtin(Value::Cell(
        CellArray::new(vec![phone, region], 1, 2).unwrap(),
    )))
    .unwrap() else {
        panic!("tensor");
    };
    assert_eq!(out.shape(), &[3, 4]);
    assert_eq!(
        out.materialize_f64(),
        vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    );
}

#[test]
fn dummyvar_treats_row_cellstr_as_one_grouping_vector() {
    let Value::Tensor(out) = block_on(dummyvar_builtin(Value::Cell(
        CellArray::new(
            vec![Value::from("red"), Value::from("blue"), Value::from("red")],
            1,
            3,
        )
        .unwrap(),
    )))
    .unwrap() else {
        panic!("tensor");
    };
    assert_eq!(out.shape(), &[3, 2]);
    assert_eq!(out.materialize_f64(), vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn dummyvar_treats_numeric_row_vector_as_one_grouping_vector() {
    let Value::Tensor(out) =
        block_on(dummyvar_builtin(tensor(vec![1.0, 2.0, 1.0], vec![1, 3]))).unwrap()
    else {
        panic!("tensor");
    };
    assert_eq!(out.shape(), &[3, 2]);
    assert_eq!(out.materialize_f64(), vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn dummyvar_preserves_rank_zero_and_rank_one_grouping_geometry() {
    let Value::Tensor(scalar) = block_on(dummyvar_builtin(tensor(vec![2.0], Vec::new()))).unwrap()
    else {
        panic!("tensor");
    };
    assert_eq!(scalar.shape(), &[1, 2]);
    assert_eq!(scalar.materialize_f64(), vec![0.0, 1.0]);

    let Value::Tensor(vector) =
        block_on(dummyvar_builtin(tensor(vec![1.0, 2.0, 1.0], vec![3]))).unwrap()
    else {
        panic!("tensor");
    };
    assert_eq!(vector.shape(), &[3, 2]);
    assert_eq!(vector.materialize_f64(), vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn dummyvar_reads_typed_integer_labels_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let Value::Tensor(out) = block_on(dummyvar_builtin(int_tensor(
        IntegerStorage::I16(vec![1, 2, 1, 2, 2, 1]),
        vec![3, 2],
    )))
    .unwrap() else {
        panic!("tensor");
    };
    assert_eq!(out.shape(), &[3, 4]);
    assert_eq!(
        out.materialize_f64(),
        vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    );
}

#[test]
fn dummyvar_supports_all_integer_classes_in_extension_mode() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let storages = [
        runmat_value::IntegerStorage::I8(vec![1, 2, 1]),
        runmat_value::IntegerStorage::I16(vec![1, 2, 1]),
        runmat_value::IntegerStorage::I32(vec![1, 2, 1]),
        runmat_value::IntegerStorage::I64(vec![1, 2, 1]),
        runmat_value::IntegerStorage::U8(vec![1, 2, 1]),
        runmat_value::IntegerStorage::U16(vec![1, 2, 1]),
        runmat_value::IntegerStorage::U32(vec![1, 2, 1]),
        runmat_value::IntegerStorage::U64(vec![1, 2, 1]),
    ];
    for storage in storages {
        let Value::Tensor(output) =
            block_on(dummyvar_builtin(int_tensor(storage, vec![3, 1]))).expect("integer groups")
        else {
            panic!("expected double tensor")
        };
        assert!(output.integer_storage().is_none());
        assert_eq!(output.shape(), &[3, 2]);
        assert_eq!(output.materialize_f64(), vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
    }
}

#[test]
fn dummyvar_rejects_wide_typed_levels_before_floating_conversion() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    for storage in [
        IntegerStorage::I64(vec![(1_i64 << 53) + 1]),
        IntegerStorage::U64(vec![(1_u64 << 53) + 1]),
        IntegerStorage::U64(vec![u64::MAX]),
    ] {
        let error = block_on(dummyvar_builtin(int_tensor(storage, vec![1, 1])))
            .expect_err("wide typed labels cannot request an unbounded design matrix");
        assert!(error.message().contains("output is too large"));
    }

    let error = block_on(dummyvar_builtin(int_tensor(
        IntegerStorage::I64(vec![i64::MIN]),
        vec![1, 1],
    )))
    .expect_err("negative typed labels remain invalid");
    assert!(error.message().contains("positive integer levels"));
}

#[test]
fn dummyvar_integer_extension_rejects_before_computation_in_strict_mode() {
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(dummyvar_builtin(int_tensor(
        runmat_value::IntegerStorage::U8(vec![1, 2]),
        vec![2, 1],
    )))
    .expect_err("integer extension gate");
    assert_eq!(
        error.identifier(),
        DUMMYVAR_INTEGER_GROUP_EXTENSION.error_identifier
    );
    drop(strict);
}

#[test]
fn dummyvar_gpu_extension_rejects_before_provider_access() {
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![2, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("genuine unregistered resident fixture")
            .handle,
    );
    let error = block_on(dummyvar_builtin(resident)).expect_err("GPU extension gate");
    assert_eq!(
        error.identifier(),
        DUMMYVAR_GPU_GROUP_EXTENSION.error_identifier
    );
    drop(strict);
}

#[test]
fn dummyvar_rejects_unrepresentable_numeric_level_before_cast() {
    let err = block_on(dummyvar_builtin(tensor(
        vec![usize::MAX as f64],
        vec![1, 1],
    )))
    .unwrap_err();
    assert!(err.message.contains("positive integer levels"));
}

#[test]
fn dummyvar_grouping_detection_uses_typed_integer_storage_len() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let first = mirrorless_int_tensor(IntegerStorage::I16(vec![1, 2, 1]), vec![3, 1]);
    let second = mirrorless_int_tensor(IntegerStorage::I16(vec![2, 2, 1]), vec![3, 1]);

    let Value::Tensor(out) = block_on(dummyvar_builtin(Value::Cell(
        CellArray::new(vec![first, second], 1, 2).unwrap(),
    )))
    .unwrap() else {
        panic!("tensor");
    };

    assert_eq!(out.shape(), &[3, 4]);
    assert_eq!(
        out.materialize_f64(),
        vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    );
}

#[test]
fn dummyvar_rejects_huge_numeric_level_before_allocating_classes() {
    let err = block_on(dummyvar_builtin(tensor(vec![1_000_000_000.0], vec![1, 1]))).unwrap_err();
    assert!(err.message.contains("output is too large"));
}

#[test]
fn dummyvar_nd_arrays_flatten_one_grouping_variable() {
    for (values, shape, expected_shape) in [
        (vec![1.0; 8], vec![2, 2, 2], vec![8, 1]),
        (vec![1.0, 2.0, 1.0], vec![1, 1, 3], vec![3, 2]),
    ] {
        let Value::Tensor(output) = block_on(dummyvar_builtin(tensor(values, shape))).unwrap()
        else {
            panic!("dummyvar must return a double matrix");
        };
        assert_eq!(output.shape(), expected_shape.as_slice());
    }
}
