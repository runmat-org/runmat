use super::super::*;
use futures::executor::block_on;

#[test]
fn bitcmp_preserves_all_native_integer_scalar_classes() {
    let cases = [
        (IntValue::I8(-11), IntValue::I8(10)),
        (IntValue::I16(-11), IntValue::I16(10)),
        (IntValue::I32(-11), IntValue::I32(10)),
        (IntValue::I64(-11), IntValue::I64(10)),
        (IntValue::U8(0b0101), IntValue::U8(0b1111_1010)),
        (IntValue::U16(0b0101), IntValue::U16(0xfffa)),
        (IntValue::U32(0b0101), IntValue::U32(0xffff_fffa)),
        (IntValue::U64(0b0101), IntValue::U64(u64::MAX - 5)),
    ];
    for (input, expected) in cases {
        let actual = block_on(bitcmp_builtin(vec![Value::Int(input)])).expect("bitcmp");
        assert_eq!(actual, Value::Int(expected));
    }
}

#[test]
fn bitcmp_preserves_exact_integer_arrays_and_default_double_behavior() {
    let input =
        Tensor::new_integer(IntegerStorage::U64(vec![0, 1_u64 << 63]), vec![1, 2]).expect("input");
    let Value::Tensor(output) =
        block_on(bitcmp_builtin(vec![Value::Tensor(input)])).expect("bitcmp")
    else {
        panic!("expected tensor result");
    };
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U64(vec![u64::MAX, !(1_u64 << 63)]))
    );

    assert_eq!(
        block_on(bitcmp_builtin(vec![Value::Num(0.0)])).expect("double bitcmp"),
        Value::Num(u64::MAX as f64)
    );
}

#[test]
fn bitcmp_sparse_double_materializes_when_complementing_implicit_zeros() {
    let sparse = runmat_value::SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![5.0, 1.0])
        .expect("sparse");
    let Value::Tensor(output) = block_on(bitcmp_builtin(vec![
        Value::SparseTensor(sparse),
        Value::String("uint8".to_string()),
    ]))
    .expect("sparse bitcmp") else {
        panic!("bitcmp must materialize a complement of sparse implicit zeros");
    };
    assert_eq!(output.shape, vec![2, 2]);
    assert_eq!(
        output.as_f64_slice().expect("double bitcmp output"),
        &[250.0, 255.0, 255.0, 254.0]
    );
}

#[test]
fn bitcmp_is_registered_and_dispatches() {
    assert!(runmat_builtins::builtin_catalog_entry_by_name(BITCMP_NAME).is_some());
    assert_eq!(
        crate::dispatcher::call_builtin(BITCMP_NAME, &[Value::Int(IntValue::U8(0b0101))])
            .expect("runtime dispatch"),
        Value::Int(IntValue::U8(0b1111_1010))
    );
}
