use super::super::*;
use futures::executor::block_on;

#[test]
fn bitget_preserves_all_native_integer_scalar_classes() {
    let cases = [
        (IntValue::I8(-1), IntValue::I8(1)),
        (IntValue::I16(-1), IntValue::I16(1)),
        (IntValue::I32(-1), IntValue::I32(1)),
        (IntValue::I64(-1), IntValue::I64(1)),
        (IntValue::U8(0b1010), IntValue::U8(0)),
        (IntValue::U16(0b1010), IntValue::U16(0)),
        (IntValue::U32(0b1010), IntValue::U32(0)),
        (IntValue::U64(0b1010), IntValue::U64(0)),
    ];
    for (input, expected) in cases {
        let actual =
            block_on(bitget_builtin(vec![Value::Int(input), Value::Num(1.0)])).expect("bitget");
        assert_eq!(actual, Value::Int(expected));
    }
}

#[test]
fn bitget_broadcasts_positions_and_preserves_uint64_storage() {
    let input =
        Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 63]), vec![1, 1]).expect("input");
    let positions = Tensor::new(vec![1.0, 63.0, 64.0], vec![1, 3]).expect("positions");
    let Value::Tensor(output) = block_on(bitget_builtin(vec![
        Value::Tensor(input),
        Value::Tensor(positions),
    ]))
    .expect("bitget") else {
        panic!("expected tensor result");
    };
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U64(vec![0, 0, 1]))
    );
}

#[test]
fn bitget_handles_signed_bits_double_output_and_invalid_positions() {
    assert_eq!(
        block_on(bitget_builtin(vec![
            Value::Int(IntValue::I8(-29)),
            Value::Num(8.0)
        ]))
        .expect("signed bit"),
        Value::Int(IntValue::I8(1))
    );
    assert_eq!(
        block_on(bitget_builtin(vec![Value::Num(8.0), Value::Num(4.0)])).expect("double bit"),
        Value::Num(1.0)
    );
    for position in [0.0, -1.0, 9.0] {
        let error = block_on(bitget_builtin(vec![
            Value::Int(IntValue::U8(1)),
            Value::Num(position),
        ]))
        .expect_err("invalid bit position");
        assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
    }
}

#[test]
fn bitget_sparse_double_scalar_position_preserves_sparse_storage() {
    let sparse = runmat_value::SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![5.0, 2.0])
        .expect("sparse");
    let Value::SparseTensor(output) = block_on(bitget_builtin(vec![
        Value::SparseTensor(sparse),
        Value::Num(1.0),
        Value::String("uint8".to_string()),
    ]))
    .expect("sparse bitget") else {
        panic!("bitget must preserve sparse storage for zero-valued implicit entries");
    };
    assert_eq!(output.shape(), vec![2, 2]);
    assert_eq!(output.get(0, 0), Some(1.0));
    assert_eq!(output.nnz(), 1);
}

#[test]
fn sparse_bitwise_position_and_value_arrays_require_matching_sizes() {
    let sparse = runmat_value::SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0, 5.0])
        .expect("sparse");

    let Value::SparseTensor(shifted) = block_on(bitshift_builtin(vec![
        Value::SparseTensor(sparse.clone()),
        Value::Tensor(Tensor::new(vec![1.0, -1.0, 1.0, -1.0], vec![2, 2]).expect("shifts")),
        Value::String("uint8".to_string()),
    ]))
    .expect("same-size sparse bitshift") else {
        panic!("bitshift leaves implicit zeros sparse for every shift");
    };
    assert_eq!(shifted.get(0, 0), Some(6.0));
    assert_eq!(shifted.get(1, 1), Some(2.0));

    let Value::SparseTensor(got) = block_on(bitget_builtin(vec![
        Value::SparseTensor(sparse.clone()),
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 1.0, 2.0], vec![2, 2]).expect("positions")),
        Value::String("uint8".to_string()),
    ]))
    .expect("same-size sparse bitget") else {
        panic!("bitget leaves implicit zeros sparse for every position");
    };
    assert_eq!(got.get(0, 0), Some(1.0));
    assert_eq!(got.nnz(), 1);

    let Value::SparseTensor(cleared) = block_on(bitset_builtin(vec![
        Value::SparseTensor(sparse.clone()),
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 1.0, 2.0], vec![2, 2]).expect("positions")),
        Value::Tensor(Tensor::new(vec![0.0; 4], vec![2, 2]).expect("clear values")),
        Value::String("uint8".to_string()),
    ]))
    .expect("same-size sparse bitset clear") else {
        panic!("clearing broadcast positions preserves implicit zeros");
    };
    assert_eq!(cleared.get(0, 0), Some(2.0));
    assert_eq!(cleared.get(1, 1), Some(5.0));

    let Value::Tensor(set) = block_on(bitset_builtin(vec![
        Value::SparseTensor(sparse),
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 1.0, 2.0], vec![2, 2]).expect("positions")),
        Value::Tensor(Tensor::new(vec![0.0, 0.0, 1.0, 1.0], vec![2, 2]).expect("set values")),
        Value::String("uint8".to_string()),
    ]))
    .expect("same-size sparse bitset set") else {
        panic!("setting an implicit position materializes the result");
    };
    assert_eq!(
        set.as_f64_slice().expect("double bitset output"),
        &[2.0, 0.0, 1.0, 7.0]
    );
}

#[test]
fn bitget_is_registered_and_dispatches() {
    assert!(runmat_builtins::builtin_catalog_entry_by_name(BITGET_NAME).is_some());
    assert_eq!(
        crate::dispatcher::call_builtin(
            BITGET_NAME,
            &[Value::Int(IntValue::U8(0b1010)), Value::Num(2.0)],
        )
        .expect("runtime dispatch"),
        Value::Int(IntValue::U8(1))
    );
}

#[test]
fn bitset_preserves_all_native_integer_scalar_classes() {
    let cases = [
        (IntValue::I8(0), IntValue::I8(2)),
        (IntValue::I16(0), IntValue::I16(2)),
        (IntValue::I32(0), IntValue::I32(2)),
        (IntValue::I64(0), IntValue::I64(2)),
        (IntValue::U8(0), IntValue::U8(2)),
        (IntValue::U16(0), IntValue::U16(2)),
        (IntValue::U32(0), IntValue::U32(2)),
        (IntValue::U64(0), IntValue::U64(2)),
    ];
    for (input, expected) in cases {
        let actual =
            block_on(bitset_builtin(vec![Value::Int(input), Value::Num(2.0)])).expect("bitset");
        assert_eq!(actual, Value::Int(expected));
    }
}

#[test]
fn bitset_supports_explicit_set_clear_and_uint64_high_bits() {
    assert_eq!(
        block_on(bitset_builtin(vec![
            Value::Int(IntValue::U8(0b1111)),
            Value::Num(3.0),
            Value::Num(0.0),
        ]))
        .expect("clear bit"),
        Value::Int(IntValue::U8(0b1011))
    );
    assert_eq!(
        block_on(bitset_builtin(vec![
            Value::Int(IntValue::I8(0)),
            Value::Num(8.0),
            Value::Bool(true),
        ]))
        .expect("set signed high bit"),
        Value::Int(IntValue::I8(i8::MIN))
    );
    assert_eq!(
        block_on(bitset_builtin(vec![
            Value::Int(IntValue::U64(0)),
            Value::Num(64.0),
            Value::Num(0.5),
        ]))
        .expect("set uint64 high bit"),
        Value::Int(IntValue::U64(1_u64 << 63))
    );
}

#[test]
fn bitset_sparse_scalar_forms_preserve_or_materialize_from_zero_semantics() {
    let sparse = runmat_value::SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0, 2.0])
        .expect("sparse");
    let Value::SparseTensor(cleared) = block_on(bitset_builtin(vec![
        Value::SparseTensor(sparse.clone()),
        Value::Num(1.0),
        Value::Num(0.0),
        Value::String("uint8".to_string()),
    ]))
    .expect("sparse clear") else {
        panic!("clearing preserves sparse storage");
    };
    assert_eq!(cleared.get(0, 0), Some(2.0));
    assert_eq!(cleared.get(1, 1), Some(2.0));

    let Value::Tensor(set) = block_on(bitset_builtin(vec![
        Value::SparseTensor(sparse),
        Value::Num(1.0),
        Value::Num(1.0),
        Value::String("uint8".to_string()),
    ]))
    .expect("sparse set") else {
        panic!("setting implicit zero materializes");
    };
    assert_eq!(
        set.as_f64_slice().expect("double bitset output"),
        &[3.0, 1.0, 1.0, 3.0]
    );
}

#[test]
fn bitset_accepts_same_size_inputs_and_scalar_values() {
    let input = Tensor::new_integer(IntegerStorage::U16(vec![0, 0]), vec![1, 2]).expect("input");
    let positions = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("positions");
    let values = Tensor::new(vec![1.0, 0.0], vec![1, 2]).expect("values");
    let Value::Tensor(output) = block_on(bitset_builtin(vec![
        Value::Tensor(input),
        Value::Tensor(positions),
        Value::Tensor(values),
    ]))
    .expect("bitset") else {
        panic!("expected tensor result");
    };
    assert_eq!(output.shape, vec![1, 2]);
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U16(vec![1, 0]))
    );

    let input = Tensor::new_integer(IntegerStorage::U16(vec![0, 0]), vec![1, 2]).expect("input");
    let Value::Tensor(output) = block_on(bitset_builtin(vec![
        Value::Tensor(input),
        Value::Num(2.0),
        Value::Bool(true),
    ]))
    .expect("scalar-expanded bitset") else {
        panic!("expected tensor result");
    };
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U16(vec![2, 2]))
    );
}

#[test]
fn bit_position_and_count_operations_reject_general_singleton_expansion() {
    let input = Tensor::new_integer(IntegerStorage::U8(vec![1, 2]), vec![2, 1]).expect("input");
    let row = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("row operand");
    for error in [
        block_on(bitshift_builtin(vec![
            Value::Tensor(input.clone()),
            Value::Tensor(row.clone()),
        ]))
        .expect_err("bitshift row/column expansion must reject"),
        block_on(bitget_builtin(vec![
            Value::Tensor(input.clone()),
            Value::Tensor(row.clone()),
        ]))
        .expect_err("bitget row/column expansion must reject"),
        block_on(bitset_builtin(vec![
            Value::Tensor(input.clone()),
            Value::Tensor(row.clone()),
            Value::Bool(true),
        ]))
        .expect_err("bitset position expansion must reject"),
        block_on(bitset_builtin(vec![
            Value::Tensor(input),
            Value::Num(1.0),
            Value::Tensor(row),
        ]))
        .expect_err("bitset value expansion must reject"),
    ] {
        assert_eq!(error.identifier(), ERROR_SIZE_MISMATCH.identifier);
        assert!(error.message().contains("exactly the same size"));
    }
}

#[test]
fn bitset_rejects_invalid_positions_nonfinite_values_and_dispatches() {
    for position in [0.0, -1.0, 9.0] {
        let error = block_on(bitset_builtin(vec![
            Value::Int(IntValue::U8(0)),
            Value::Num(position),
        ]))
        .expect_err("invalid position");
        assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
    }
    let error = block_on(bitset_builtin(vec![
        Value::Int(IntValue::U8(0)),
        Value::Num(1.0),
        Value::Num(f64::NAN),
    ]))
    .expect_err("nonfinite bit value");
    assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);

    assert!(runmat_builtins::builtin_catalog_entry_by_name(BITSET_NAME).is_some());
    assert_eq!(
        crate::dispatcher::call_builtin(
            BITSET_NAME,
            &[Value::Int(IntValue::U8(0)), Value::Num(1.0)],
        )
        .expect("runtime dispatch"),
        Value::Int(IntValue::U8(1))
    );
}
