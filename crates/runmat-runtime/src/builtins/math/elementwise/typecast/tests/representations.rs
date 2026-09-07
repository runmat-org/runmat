use super::super::typecast_builtin;
use futures::executor::block_on;
use runmat_value::{IntegerStorage, LogicalArray, NumericStorage, Tensor, Value};

fn round_trip(storage: NumericStorage, class: &str) {
    let input = Tensor::from_numeric_storage(storage.clone(), vec![1, 2]).expect("input");
    let bytes = block_on(typecast_builtin(vec![
        Value::Tensor(input),
        Value::String("uint8".into()),
    ]))
    .expect("encode");
    let Value::Tensor(output) =
        block_on(typecast_builtin(vec![bytes, Value::String(class.into())])).expect("decode")
    else {
        panic!("expected tensor");
    };
    assert_eq!(output.shape, vec![1, 2], "{class}");
    assert_eq!(
        output.into_numeric_storage().expect("output storage"),
        storage,
        "{class}"
    );
}

#[test]
fn every_numeric_storage_class_round_trips_through_bytes() {
    round_trip(NumericStorage::F64(vec![-1.25, 9.5]), "double");
    round_trip(NumericStorage::F32(vec![-1.25, 9.5]), "single");
    round_trip(NumericStorage::I8(vec![i8::MIN, i8::MAX]), "int8");
    round_trip(NumericStorage::I16(vec![i16::MIN, i16::MAX]), "int16");
    round_trip(NumericStorage::I32(vec![i32::MIN, i32::MAX]), "int32");
    round_trip(NumericStorage::I64(vec![i64::MIN, i64::MAX]), "int64");
    round_trip(NumericStorage::U8(vec![0, u8::MAX]), "uint8");
    round_trip(NumericStorage::U16(vec![0, u16::MAX]), "uint16");
    round_trip(NumericStorage::U32(vec![0, u32::MAX]), "uint32");
    round_trip(NumericStorage::U64(vec![0, u64::MAX]), "uint64");
}

#[test]
fn logical_target_normalizes_each_source_byte() {
    let input =
        Tensor::new_integer(IntegerStorage::U8(vec![0, 1, 2, 255]), vec![1, 4]).expect("input");
    let Value::LogicalArray(output) = block_on(typecast_builtin(vec![
        Value::Tensor(input),
        Value::String("logical".into()),
    ]))
    .expect("logical output") else {
        panic!("expected logical array");
    };
    assert_eq!(&*output.data, &[0, 1, 1, 1]);
}

#[test]
fn logical_source_exposes_its_normalized_byte_storage() {
    let input = LogicalArray::new(vec![0, 1, 1], vec![3, 1]).expect("input");
    let Value::Tensor(output) = block_on(typecast_builtin(vec![
        Value::LogicalArray(input),
        Value::String("uint8".into()),
    ]))
    .expect("numeric output") else {
        panic!("expected numeric tensor");
    };
    assert_eq!(output.shape, vec![3, 1]);
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U8(vec![0, 1, 1]))
    );
}

#[test]
fn empty_vectors_retain_orientation() {
    for (shape, expected) in [(vec![1, 0], vec![1, 0]), (vec![0, 1], vec![0, 1])] {
        let input = Tensor::new_integer(IntegerStorage::U32(vec![]), shape).expect("input");
        let Value::Tensor(output) = block_on(typecast_builtin(vec![
            Value::Tensor(input),
            Value::String("uint8".into()),
        ]))
        .expect("empty output") else {
            panic!("expected tensor");
        };
        assert_eq!(output.shape, expected);
    }
}
