use super::*;

#[tokio::test]
async fn topk_k_uses_exact_storage_for_every_integer_class_and_rejects_wide_values() {
    let storages = vec![
        IntegerStorage::I8(vec![1]),
        IntegerStorage::I16(vec![1]),
        IntegerStorage::I32(vec![1]),
        IntegerStorage::I64(vec![1]),
        IntegerStorage::U8(vec![1]),
        IntegerStorage::U16(vec![1]),
        IntegerStorage::U32(vec![1]),
        IntegerStorage::U64(vec![1]),
    ];
    for storage in storages {
        let k = Tensor::new_integer(storage, vec![1, 1]).expect("k");
        assert_eq!(parse_k(TopKKind::Max, &Value::Tensor(k)).await.unwrap(), 1);
    }
    let wide =
        Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("wide k");
    if usize::BITS == 64 {
        assert_eq!(
            parse_k(TopKKind::Max, &Value::Tensor(wide)).await.unwrap(),
            usize::MAX
        );
    } else {
        assert!(parse_k(TopKKind::Max, &Value::Tensor(wide)).await.is_err());
    }
}

#[tokio::test]
async fn topk_dimension_uses_exact_storage_for_every_integer_class() {
    let storages = vec![
        IntegerStorage::I8(vec![2]),
        IntegerStorage::I16(vec![2]),
        IntegerStorage::I32(vec![2]),
        IntegerStorage::I64(vec![2]),
        IntegerStorage::U8(vec![2]),
        IntegerStorage::U16(vec![2]),
        IntegerStorage::U32(vec![2]),
        IntegerStorage::U64(vec![2]),
    ];
    for storage in storages {
        let dim = Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).expect("dim"));
        let input = tensor(vec![3.0, 4.0, 1.0, 2.0], vec![2, 2]);
        let evaluation = evaluate_topk(
            TopKKind::Min,
            input,
            crate::call::FunctionArgumentView::from_slice(&[Value::Num(1.0), dim]),
        )
        .await
        .expect("typed dimension");
        let Value::Tensor(values) = evaluation.values else {
            panic!("expected tensor");
        };
        assert_eq!(values.shape(), vec![2, 1]);
        assert_eq!(self::values(&values), vec![1.0, 2.0]);
    }
}
