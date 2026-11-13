use super::*;
use futures::executor::block_on;
use runmat_value::{IntValue, IntegerComplexStorage, IntegerStorage};

#[test]
fn empty_and_nd_vectors_preserve_geometry() {
    for shape in [vec![0, 3], vec![1, 0, 1]] {
        let input = Tensor::new(vec![], shape).unwrap();
        let Value::Tensor(output) = block_on(toeplitz_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(input),
                row(&[1., 2.]),
            ]),
        ))
        .unwrap() else {
            panic!("tensor");
        };
        assert_eq!(output.shape(), [0, 2]);
    }
    let input = Tensor::new(vec![1., 2., 3.], vec![1, 1, 3]).unwrap();
    let Value::Tensor(output) = block_on(toeplitz_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(input)]),
    ))
    .unwrap() else {
        panic!("tensor");
    };
    assert_eq!(output.shape(), [3, 3]);
}

#[test]
fn output_count_and_arity_fail_before_provider_lookup() {
    let _count = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let error = block_on(toeplitz_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        runmat_builtins::TOEPLITZ_ERROR_TOO_MANY_OUTPUTS.identifier
    );
    drop(_count);
    assert_eq!(
        block_on(toeplitz_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![])
        ))
        .unwrap_err()
        .identifier(),
        ERROR_INVALID_INPUT.identifier
    );
    let _discard = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    assert!(block_on(toeplitz_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![row(&[1., 2.])])
    ))
    .is_ok());
}

#[test]
fn resident_exact_integer_construction_preserves_values_and_owner() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![1, 2],
        )
        .unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor)
            .unwrap()
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("validated fixture provenance");
        let Value::GpuTensor(output) = block_on(toeplitz_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                handle.clone(),
            )]),
        ))
        .unwrap() else {
            panic!("resident output");
        };
        assert_eq!(output.device_id(), handle.device_id());
        let Value::Tensor(host) = block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(
            output.clone(),
        )))
        .unwrap() else {
            panic!("host tensor");
        };
        assert_eq!(
            host.integer_storage(),
            Some(&IntegerStorage::U64(vec![
                9_007_199_254_740_993,
                u64::MAX,
                u64::MAX,
                9_007_199_254_740_993
            ]))
        );
        drop(output);
        drop(handle);
    });
}

fn row(values: &[f64]) -> Value {
    Value::Tensor(Tensor::new(values.to_vec(), vec![1, values.len()]).expect("tensor"))
}

#[test]
fn toeplitz_one_real_vector_is_symmetric() {
    let out = block_on(toeplitz_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![row(&[1.0, 2.0, 3.0])]),
    ))
    .expect("toeplitz");
    let Value::Tensor(tensor) = out else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.shape(), [3, 3]);
    assert_eq!(
        tensor.materialize_f64(),
        vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]
    );
}

#[test]
fn toeplitz_column_and_row_use_first_column_diagonal() {
    let out = block_on(toeplitz_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            row(&[1.0, 2.0, 3.0]),
            row(&[9.0, 4.0]),
        ]),
    ))
    .expect("toeplitz");
    let Value::Tensor(tensor) = out else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.shape(), [3, 2]);
    assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0]);
}

#[test]
fn toeplitz_complex_single_input_uses_row_and_conjugates_first_column() {
    let r = ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![1, 2]).unwrap();
    let out = block_on(toeplitz_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::ComplexTensor(r)]),
    ))
    .expect("toeplitz");
    let Value::ComplexTensor(tensor) = out else {
        panic!("expected complex tensor");
    };
    assert_eq!(tensor.shape(), [2, 2]);
    assert_eq!(
        tensor.materialize_f64(),
        vec![(1.0, 2.0), (3.0, -4.0), (3.0, 4.0), (1.0, 2.0)]
    );
}

#[test]
fn toeplitz_preserves_native_single_storage() {
    let input = Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let out = block_on(toeplitz_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(input)]),
    ))
    .expect("toeplitz");
    let Value::Tensor(tensor) = out else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.shape(), [3, 3]);
    assert_eq!(
        tensor.into_numeric_storage().expect("single storage"),
        NumericStorage::F32(vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0])
    );
}

#[test]
fn toeplitz_preserves_complex_single_storage() {
    let input = ComplexTensor::from_f32(vec![(1.0, 2.0), (3.0, 4.0)], vec![1, 2]).unwrap();
    let Value::ComplexTensor(output) = block_on(toeplitz_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::ComplexTensor(input)]),
    ))
    .unwrap() else {
        panic!("expected complex tensor")
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
}

#[test]
fn toeplitz_rejects_mixed_integer_classes_without_floating_conversion() {
    let column = Tensor::new_integer(IntegerStorage::I64(vec![1, i64::MAX]), vec![2, 1]).unwrap();
    let row = Tensor::new_integer(IntegerStorage::U64(vec![1, u64::MAX]), vec![1, 2]).unwrap();
    let error = block_on(toeplitz_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(column),
            Value::Tensor(row),
        ]),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
}

#[test]
fn toeplitz_preserves_all_exact_integer_classes() {
    let storages = [
        IntegerStorage::I8(vec![-2, 7, 9]),
        IntegerStorage::I16(vec![-300, 400, 900]),
        IntegerStorage::I32(vec![i32::MIN, 0, i32::MAX]),
        IntegerStorage::I64(vec![i64::MIN, 0, i64::MAX]),
        IntegerStorage::U8(vec![0, 7, u8::MAX]),
        IntegerStorage::U16(vec![0, 700, u16::MAX]),
        IntegerStorage::U32(vec![0, 9_007_199, u32::MAX]),
        IntegerStorage::U64(vec![0, 9_007_199_254_740_993, u64::MAX]),
    ];

    for storage in storages {
        let values = storage.exact_values();
        let expected = storage
            .from_exact_values_like(vec![
                values[0].clone(),
                values[1].clone(),
                values[2].clone(),
                values[1].clone(),
                values[0].clone(),
                values[1].clone(),
                values[2].clone(),
                values[1].clone(),
                values[0].clone(),
            ])
            .expect("expected toeplitz storage");
        let input = Tensor::new_integer(storage, vec![1, 3]).expect("integer vector");
        let Value::Tensor(output) = block_on(toeplitz_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(input)]),
        ))
        .expect("toeplitz") else {
            panic!("expected exact integer tensor");
        };
        assert_eq!(output.shape(), [3, 3]);
        assert_eq!(output.integer_storage(), Some(&expected));
    }

    let column =
        Tensor::new_integer(IntegerStorage::U64(vec![7, u64::MAX]), vec![2, 1]).expect("column");
    let row = Tensor::new_integer(
        IntegerStorage::U64(vec![7, 9_007_199_254_740_993]),
        vec![1, 2],
    )
    .expect("row");
    let Value::Tensor(output) = block_on(toeplitz_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(column),
            Value::Tensor(row),
        ]),
    ))
    .expect("toeplitz") else {
        panic!("expected exact integer tensor");
    };
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U64(vec![
            7,
            u64::MAX,
            9_007_199_254_740_993,
            7
        ]))
    );

    let Value::Tensor(scalar) = block_on(toeplitz_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(IntValue::U64(
            u64::MAX,
        ))]),
    ))
    .expect("scalar toeplitz") else {
        panic!("expected exact scalar tensor");
    };
    assert_eq!(
        scalar.integer_storage(),
        Some(&IntegerStorage::U64(vec![u64::MAX]))
    );
}

#[test]
fn toeplitz_typed_complex_integer_vector_shape_uses_storage_len_not_mirror() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let storage = IntegerComplexStorage::new(
        IntegerStorage::I16(vec![1, 2, 3]),
        IntegerStorage::I16(vec![4, 5, 6]),
    )
    .expect("complex integer storage");
    let input = ComplexTensor::new_integer(storage, vec![1, 3]).expect("complex vector");

    let Value::ComplexTensor(output) = block_on(toeplitz_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::ComplexTensor(input)]),
    ))
    .expect("toeplitz") else {
        panic!("expected exact complex integer tensor");
    };

    assert_eq!(output.shape(), [3, 3]);
    let storage = output.integer_storage().expect("typed complex storage");
    assert_eq!(
        storage.real(),
        &IntegerStorage::I16(vec![1, 2, 3, 2, 1, 2, 3, 2, 1])
    );
    assert_eq!(
        storage.imag(),
        &IntegerStorage::I16(vec![4, -5, -6, 5, 4, -5, 6, 5, 4])
    );
}

#[test]
fn toeplitz_complex_integer_requires_runmat_extensions() {
    let storage = IntegerComplexStorage::new(
        IntegerStorage::I16(vec![1, 2]),
        IntegerStorage::I16(vec![3, 4]),
    )
    .expect("complex integer storage");
    let input = ComplexTensor::new_integer(storage, vec![1, 2]).expect("complex vector");
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(toeplitz_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::ComplexTensor(input)]),
    ))
    .expect_err("MATLAB mode rejects typed complex integer input");
    assert_eq!(
        error.identifier(),
        COMPLEX_INTEGER_INPUT_EXTENSION.error_identifier
    );
}

#[test]
fn toeplitz_rejects_matrix_inputs() {
    let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let err = block_on(toeplitz_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(matrix)]),
    ))
    .expect_err("matrix input should fail");
    assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
}
