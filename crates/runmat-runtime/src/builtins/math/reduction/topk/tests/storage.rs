use super::*;

#[tokio::test]
async fn topk_rejects_typed_complex_integer_inputs() {
    let input = Value::ComplexTensor(
        ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::I64(vec![i64::MAX]),
                IntegerStorage::I64(vec![-1]),
            )
            .expect("storage"),
            vec![1, 1],
        )
        .expect("tensor"),
    );
    let err = evaluate_topk(
        TopKKind::Max,
        input,
        crate::call::FunctionArgumentView::from_slice(&[Value::Num(1.0)]),
    )
    .await
    .expect_err("typed complex integer input must reject");
    assert!(err.message().contains("complex numbers with integer types"));
}

#[tokio::test]
async fn topk_preserves_native_single_storage() {
    let input =
        Value::Tensor(Tensor::from_f32(vec![1.0, 3.0, 2.0], vec![3, 1]).expect("single input"));
    let evaluation = evaluate_topk(
        TopKKind::Max,
        input,
        crate::call::FunctionArgumentView::from_slice(&[Value::Num(2.0)]),
    )
    .await
    .expect("topk");
    let Value::Tensor(output) = evaluation.values else {
        panic!("expected single tensor");
    };
    assert_eq!(
        output.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![3.0, 2.0])
    );
}

#[tokio::test]
async fn topk_preserves_native_complex_single_and_logical_storage() {
    let complex = Value::ComplexTensor(
        ComplexTensor::from_f32(vec![(1.0, 1.0), (3.0, -1.0), (2.0, 0.0)], vec![3, 1])
            .expect("complex single"),
    );
    let evaluation = evaluate_topk(
        TopKKind::Max,
        complex,
        crate::call::FunctionArgumentView::from_slice(&[
            Value::Num(2.0),
            Value::from("ComparisonMethod"),
            Value::from("real"),
        ]),
    )
    .await
    .expect("complex topk");
    let Value::ComplexTensor(values) = evaluation.values else {
        panic!("expected complex tensor");
    };
    assert_eq!(
        values.into_complex_storage(),
        runmat_value::ComplexStorage::F32(vec![(3.0, -1.0), (2.0, 0.0)].into())
    );

    let logical =
        Value::LogicalArray(LogicalArray::new(vec![0, 1, 0], vec![3, 1]).expect("logical input"));
    let evaluation = evaluate_topk(
        TopKKind::Max,
        logical,
        crate::call::FunctionArgumentView::from_slice(&[Value::Num(2.0)]),
    )
    .await
    .expect("logical topk");
    let Value::LogicalArray(values) = evaluation.values else {
        panic!("expected logical output");
    };
    assert_eq!(values.shape, vec![2, 1]);
    assert_eq!(values.data, vec![1, 0]);
    let Value::Tensor(indices) = evaluation.indices else {
        panic!("expected indices");
    };
    assert_eq!(self::values(&indices), vec![2.0, 1.0]);
}

#[tokio::test]
async fn topk_preserves_wide_integer_scalar_exactly() {
    let evaluation = evaluate_topk(
        TopKKind::Max,
        Value::Int(IntValue::U64(u64::MAX)),
        crate::call::FunctionArgumentView::from_slice(&[Value::Num(1.0)]),
    )
    .await
    .expect("topk");
    assert_eq!(evaluation.values, Value::Int(IntValue::U64(u64::MAX)));
    assert_eq!(evaluation.indices, Value::Num(1.0));
}
