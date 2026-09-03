use super::super::*;
use futures::executor::block_on;

#[test]
fn bitwise_uint32_scalars_preserve_uint32() {
    let out = block_on(bitor_builtin(vec![
        Value::Int(IntValue::U32(0b0101)),
        Value::Int(IntValue::U32(0b0011)),
    ]))
    .expect("bitor");
    assert_eq!(out, Value::Int(IntValue::U32(0b0111)));
}

#[test]
fn bitand_broadcasts_tensor_and_scalar() {
    let tensor =
        Tensor::new_with_dtype(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], NumericDType::U32).unwrap();
    let out = block_on(bitand_builtin(vec![
        Value::Tensor(tensor),
        Value::Int(IntValue::U32(1)),
    ]))
    .expect("bitand");
    match out {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            assert_eq!(
                t.integer_storage(),
                Some(&IntegerStorage::U32(vec![1, 0, 1, 0]))
            );
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn binary_bitwise_sparse_operands_preserve_or_materialize_by_zero_semantics() {
    let left = runmat_value::SparseTensor::new(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![0b0110 as f64, 0b1010 as f64],
    )
    .expect("left sparse");
    let right = runmat_value::SparseTensor::new(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 0],
        vec![0b0011 as f64, 0b0101 as f64],
    )
    .expect("right sparse");

    let Value::SparseTensor(and) = block_on(bitand_builtin(vec![
        Value::SparseTensor(left.clone()),
        Value::SparseTensor(right.clone()),
        Value::String("uint8".to_string()),
    ]))
    .expect("sparse bitand") else {
        panic!("bitand of sparse operands preserves CSC storage");
    };
    assert_eq!(and.get(0, 0), Some(2.0));
    assert_eq!(and.nnz(), 1);

    let Value::SparseTensor(or) = block_on(bitor_builtin(vec![
        Value::SparseTensor(left.clone()),
        Value::SparseTensor(right),
        Value::String("uint8".to_string()),
    ]))
    .expect("sparse bitor") else {
        panic!("bitor of sparse operands preserves CSC storage");
    };
    assert_eq!(or.get(0, 0), Some(7.0));
    assert_eq!(or.get(1, 1), Some(10.0));
    assert_eq!(or.nnz(), 3);

    let Value::Tensor(xor) = block_on(bitxor_builtin(vec![
        Value::SparseTensor(left),
        Value::Num(1.0),
        Value::String("uint8".to_string()),
    ]))
    .expect("sparse xor nonzero scalar") else {
        panic!("xor with a nonzero scalar materializes implicit zeros");
    };
    assert_eq!(
        xor.as_f64_slice().expect("double bitxor output"),
        &[7.0, 1.0, 1.0, 11.0]
    );
}

#[test]
fn binary_bitwise_sparse_dense_and_broadcast_forms_use_zero_aware_output_storage() {
    let sparse = runmat_value::SparseTensor::new(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![0b0110 as f64, 0b1010 as f64],
    )
    .expect("sparse");
    let dense =
        Tensor::new(vec![0b0011 as f64, 0.0, 0.0, 0b1111 as f64], vec![2, 2]).expect("dense");
    let Value::SparseTensor(and) = block_on(bitand_builtin(vec![
        Value::SparseTensor(sparse.clone()),
        Value::Tensor(dense),
        Value::String("uint8".to_string()),
    ]))
    .expect("sparse dense bitand") else {
        panic!("bitand keeps sparse output because zero is annihilating");
    };
    assert_eq!(and.get(0, 0), Some(2.0));
    assert_eq!(and.get(1, 1), Some(10.0));
    assert_eq!(and.nnz(), 2);

    let Value::Tensor(or) = block_on(bitor_builtin(vec![
        Value::SparseTensor(sparse),
        Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).expect("broadcast dense")),
        Value::String("uint8".to_string()),
    ]))
    .expect("sparse dense bitor") else {
        panic!("bitor materializes when broadcast dense values make implicit zeros nonzero");
    };
    assert_eq!(or.shape, vec![2, 2]);
    assert_eq!(
        or.as_f64_slice().expect("double bitor output"),
        &[6.0, 1.0, 0.0, 11.0]
    );
}

#[test]
fn binary_bitwise_rejects_runmat_typed_sparse_integer_extension() {
    let typed = runmat_value::SparseTensor::new_integer(
        1,
        1,
        vec![0, 1],
        vec![0],
        IntegerStorage::U8(vec![1]),
    )
    .expect("typed sparse");
    let error = block_on(bitand_builtin(vec![
        Value::SparseTensor(typed),
        Value::Num(1.0),
    ]))
    .expect_err("typed sparse is not a MATLAB sparse integer representation");
    assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
}

#[test]
fn binary_bitwise_rejects_mixed_integer_classes() {
    let forward = block_on(bitor_builtin(vec![
        Value::Int(IntValue::U8(1)),
        Value::Int(IntValue::U32(256)),
    ]))
    .expect_err("mixed integer classes must fail");
    let reverse = block_on(bitor_builtin(vec![
        Value::Int(IntValue::U32(256)),
        Value::Int(IntValue::U8(1)),
    ]))
    .expect_err("mixed integer classes must fail");

    assert_eq!(forward.identifier(), ERROR_INVALID_INPUT.identifier);
    assert_eq!(reverse.identifier(), ERROR_INVALID_INPUT.identifier);
}

#[test]
fn bitwise_preserves_all_native_integer_scalar_classes() {
    let cases = [
        (IntValue::I8(-5), IntValue::I8(6), IntValue::I8(2)),
        (IntValue::I16(-5), IntValue::I16(6), IntValue::I16(2)),
        (IntValue::I32(-5), IntValue::I32(6), IntValue::I32(2)),
        (IntValue::I64(-5), IntValue::I64(6), IntValue::I64(2)),
        (
            IntValue::U8(0b1010),
            IntValue::U8(0b0110),
            IntValue::U8(0b0010),
        ),
        (
            IntValue::U16(0b1010),
            IntValue::U16(0b0110),
            IntValue::U16(0b0010),
        ),
        (
            IntValue::U32(0b1010),
            IntValue::U32(0b0110),
            IntValue::U32(0b0010),
        ),
        (
            IntValue::U64(0b1010),
            IntValue::U64(0b0110),
            IntValue::U64(0b0010),
        ),
    ];
    for (left, right, expected) in cases {
        let actual =
            block_on(bitand_builtin(vec![Value::Int(left), Value::Int(right)])).expect("bitand");
        assert_eq!(actual, Value::Int(expected));
    }

    let high = block_on(bitor_builtin(vec![
        Value::Int(IntValue::U64(1_u64 << 63)),
        Value::Int(IntValue::U64(1_u64 << 60)),
    ]))
    .expect("uint64 high-bit bitor");
    assert_eq!(
        high,
        Value::Int(IntValue::U64((1_u64 << 63) | (1_u64 << 60)))
    );
}

#[test]
fn bitxor_preserves_all_native_integer_scalar_classes() {
    let cases = [
        (IntValue::I8(-5), IntValue::I8(6), IntValue::I8(-3)),
        (IntValue::I16(-5), IntValue::I16(6), IntValue::I16(-3)),
        (IntValue::I32(-5), IntValue::I32(6), IntValue::I32(-3)),
        (IntValue::I64(-5), IntValue::I64(6), IntValue::I64(-3)),
        (
            IntValue::U8(0b1010),
            IntValue::U8(0b0110),
            IntValue::U8(0b1100),
        ),
        (
            IntValue::U16(0b1010),
            IntValue::U16(0b0110),
            IntValue::U16(0b1100),
        ),
        (
            IntValue::U32(0b1010),
            IntValue::U32(0b0110),
            IntValue::U32(0b1100),
        ),
        (
            IntValue::U64(0b1010),
            IntValue::U64(0b0110),
            IntValue::U64(0b1100),
        ),
    ];
    for (left, right, expected) in cases {
        let actual =
            block_on(bitxor_builtin(vec![Value::Int(left), Value::Int(right)])).expect("bitxor");
        assert_eq!(actual, Value::Int(expected));
    }
}

#[test]
fn bitxor_broadcasts_exact_uint64_storage_without_losing_high_bits() {
    let left = Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]), vec![1, 2])
        .expect("left");
    let Value::Tensor(output) = block_on(bitxor_builtin(vec![
        Value::Tensor(left),
        Value::Int(IntValue::U64(1_u64 << 60)),
    ]))
    .expect("bitxor") else {
        panic!("expected tensor result");
    };
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U64(vec![
            (1_u64 << 63) | (1_u64 << 60),
            u64::MAX ^ (1_u64 << 60),
        ]))
    );
}

#[test]
fn bitxor_follows_binary_integer_class_rules_and_is_registered() {
    assert!(runmat_builtins::builtin_catalog_entry_by_name(BITXOR_NAME).is_some());
    assert_eq!(
        crate::dispatcher::call_builtin(
            BITXOR_NAME,
            &[
                Value::Int(IntValue::U8(0b1010)),
                Value::Int(IntValue::U8(0b0110))
            ],
        )
        .expect("runtime dispatch"),
        Value::Int(IntValue::U8(0b1100))
    );
    assert_eq!(
        block_on(bitxor_builtin(vec![
            Value::Int(IntValue::U16(0b1010)),
            Value::Num(6.0),
        ]))
        .expect("scalar double bitxor"),
        Value::Int(IntValue::U16(0b1100))
    );
    let error = block_on(bitxor_builtin(vec![
        Value::Int(IntValue::U8(1)),
        Value::Int(IntValue::U16(1)),
    ]))
    .expect_err("mixed integer classes must fail");
    assert_eq!(error.identifier(), ERROR_INVALID_INPUT.identifier);
}

#[test]
fn bitwise_native_integer_arrays_preserve_exact_64_bit_storage() {
    let left = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]), vec![1, 2])
        .expect("left");
    let right = Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 60, u64::MAX]), vec![1, 2])
        .expect("right");
    let Value::Tensor(output) = block_on(bitand_builtin(vec![
        Value::Tensor(left),
        Value::Tensor(right),
    ]))
    .expect("bitand") else {
        panic!("expected tensor result");
    };
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U64(vec![1_u64 << 60, 1_u64 << 63]))
    );
}
