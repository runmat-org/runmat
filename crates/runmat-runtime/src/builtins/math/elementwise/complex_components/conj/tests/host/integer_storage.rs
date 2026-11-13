use super::*;

#[test]
fn conj_typed_integer_imaginary_storage_preserves_class_and_saturates() {
    let cases = [
        (
            IntegerStorage::I8(vec![3, i8::MIN]),
            IntegerStorage::I8(vec![-3, i8::MAX]),
        ),
        (
            IntegerStorage::I16(vec![3, i16::MIN]),
            IntegerStorage::I16(vec![-3, i16::MAX]),
        ),
        (
            IntegerStorage::I32(vec![3, i32::MIN]),
            IntegerStorage::I32(vec![-3, i32::MAX]),
        ),
        (
            IntegerStorage::I64(vec![3, i64::MIN]),
            IntegerStorage::I64(vec![-3, i64::MAX]),
        ),
        (
            IntegerStorage::U8(vec![3, u8::MAX]),
            IntegerStorage::U8(vec![0, 0]),
        ),
        (
            IntegerStorage::U16(vec![3, u16::MAX]),
            IntegerStorage::U16(vec![0, 0]),
        ),
        (
            IntegerStorage::U32(vec![3, u32::MAX]),
            IntegerStorage::U32(vec![0, 0]),
        ),
        (
            IntegerStorage::U64(vec![3, u64::MAX]),
            IntegerStorage::U64(vec![0, 0]),
        ),
    ];
    for (input, expected) in cases {
        assert_eq!(conjugate_integer_imaginary_storage(input), expected);
    }
}
