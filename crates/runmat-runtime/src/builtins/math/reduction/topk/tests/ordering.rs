use super::*;

#[tokio::test]
async fn topk_abs_comparison_handles_signed_min_without_overflow() {
    let input = Value::Tensor(
        Tensor::new_integer(IntegerStorage::I64(vec![i64::MAX, i64::MIN]), vec![2, 1])
            .expect("integer input"),
    );
    let evaluation = evaluate_topk(
        TopKKind::Max,
        input,
        crate::call::FunctionArgumentView::from_slice(&[
            Value::Num(1.0),
            Value::from("ComparisonMethod"),
            Value::from("abs"),
        ]),
    )
    .await
    .expect("topk");
    assert_eq!(evaluation.values, Value::Int(IntValue::I64(i64::MIN)));
    assert_eq!(evaluation.indices, Value::Num(2.0));
}

#[tokio::test]
async fn topk_abs_comparison_uses_phase_for_equal_real_magnitudes() {
    for (kind, expected_values, expected_indices) in [
        (TopKKind::Max, vec![-3.0, 3.0], vec![1.0, 2.0]),
        (TopKKind::Min, vec![3.0, -3.0], vec![2.0, 1.0]),
    ] {
        let evaluation = evaluate_topk(
            kind,
            tensor(vec![-3.0, 3.0], vec![2, 1]),
            crate::call::FunctionArgumentView::from_slice(&[
                Value::Num(2.0),
                Value::from("ComparisonMethod"),
                Value::from("abs"),
            ]),
        )
        .await
        .expect("absolute topk");
        let Value::Tensor(values) = evaluation.values else {
            panic!("expected values");
        };
        assert_eq!(self::values(&values), expected_values);
        let Value::Tensor(indices) = evaluation.indices else {
            panic!("expected indices");
        };
        assert_eq!(self::values(&indices), expected_indices);
    }
}

#[tokio::test]
async fn maxk_real_abs_comparison_uses_magnitude() {
    let input = tensor(vec![-3.0, 2.0], vec![1, 2]);
    let eval = evaluate_topk(
        TopKKind::Max,
        input,
        crate::call::FunctionArgumentView::from_slice(&[
            Value::Num(1.0),
            Value::from("ComparisonMethod"),
            Value::from("abs"),
        ]),
    )
    .await
    .unwrap();
    assert_eq!(eval.values, Value::Num(-3.0));
    assert_eq!(eval.indices, Value::Num(1.0));
}

#[tokio::test]
async fn maxk_supports_complex_comparison_method_real() {
    let input = Value::ComplexTensor(
        ComplexTensor::new(vec![(1.0, 10.0), (3.0, -1.0), (2.0, 0.0)], vec![3, 1]).unwrap(),
    );
    let eval = evaluate_topk(
        TopKKind::Max,
        input,
        crate::call::FunctionArgumentView::from_slice(&[
            Value::Num(2.0),
            Value::from("ComparisonMethod"),
            Value::from("real"),
        ]),
    )
    .await
    .unwrap();
    let Value::ComplexTensor(values) = eval.values else {
        panic!("expected complex tensor");
    };
    assert_eq!(values.materialize_f64(), vec![(3.0, -1.0), (2.0, 0.0)]);
}
