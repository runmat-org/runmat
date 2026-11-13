use super::*;

#[tokio::test]
async fn topk_preserves_every_integer_class_and_returns_double_indices() {
    let cases = vec![
        (
            IntegerStorage::I8(vec![i8::MIN, 0, i8::MAX]),
            NumericStorage::I8(vec![i8::MAX, 0]),
            NumericStorage::I8(vec![i8::MIN, 0]),
        ),
        (
            IntegerStorage::I16(vec![i16::MIN, 0, i16::MAX]),
            NumericStorage::I16(vec![i16::MAX, 0]),
            NumericStorage::I16(vec![i16::MIN, 0]),
        ),
        (
            IntegerStorage::I32(vec![i32::MIN, 0, i32::MAX]),
            NumericStorage::I32(vec![i32::MAX, 0]),
            NumericStorage::I32(vec![i32::MIN, 0]),
        ),
        (
            IntegerStorage::I64(vec![i64::MIN, 0, i64::MAX]),
            NumericStorage::I64(vec![i64::MAX, 0]),
            NumericStorage::I64(vec![i64::MIN, 0]),
        ),
        (
            IntegerStorage::U8(vec![0, u8::MAX - 1, u8::MAX]),
            NumericStorage::U8(vec![u8::MAX, u8::MAX - 1]),
            NumericStorage::U8(vec![0, u8::MAX - 1]),
        ),
        (
            IntegerStorage::U16(vec![0, u16::MAX - 1, u16::MAX]),
            NumericStorage::U16(vec![u16::MAX, u16::MAX - 1]),
            NumericStorage::U16(vec![0, u16::MAX - 1]),
        ),
        (
            IntegerStorage::U32(vec![0, u32::MAX - 1, u32::MAX]),
            NumericStorage::U32(vec![u32::MAX, u32::MAX - 1]),
            NumericStorage::U32(vec![0, u32::MAX - 1]),
        ),
        (
            IntegerStorage::U64(vec![0, u64::MAX - 1, u64::MAX]),
            NumericStorage::U64(vec![u64::MAX, u64::MAX - 1]),
            NumericStorage::U64(vec![0, u64::MAX - 1]),
        ),
    ];
    for (input, expected_max, expected_min) in cases {
        for (kind, expected_values, expected_indices) in [
            (TopKKind::Max, expected_max, vec![3.0, 2.0]),
            (TopKKind::Min, expected_min, vec![1.0, 2.0]),
        ] {
            let input = Value::Tensor(
                Tensor::new_integer(input.clone(), vec![3, 1]).expect("integer input"),
            );
            let evaluation = evaluate_topk(
                kind,
                input,
                crate::call::FunctionArgumentView::from_slice(&[Value::Num(2.0)]),
            )
            .await
            .expect("topk");
            let Value::Tensor(output) = evaluation.values else {
                panic!("expected typed tensor output");
            };
            assert_eq!(output.into_numeric_storage().unwrap(), expected_values);
            let Value::Tensor(indices) = evaluation.indices else {
                panic!("expected double index tensor");
            };
            assert_eq!(values(&indices), expected_indices);
            assert!(matches!(
                indices.into_numeric_storage().unwrap(),
                NumericStorage::F64(_)
            ));
        }
    }
}
