use super::*;

#[tokio::test]
async fn zero_k_on_trailing_dimension_creates_an_empty_axis() {
    for kind in [TopKKind::Max, TopKKind::Min] {
        for input in [
            tensor(vec![1., 2.], vec![2, 1]),
            Value::ComplexTensor(ComplexTensor::new(vec![(1., 2.), (3., 4.)], vec![2, 1]).unwrap()),
        ] {
            let result = evaluate_topk(kind, input, &[Value::Num(0.), Value::Num(3.)])
                .await
                .unwrap();
            let shape = match result.values {
                Value::Tensor(value) => value.shape,
                Value::ComplexTensor(value) => value.shape,
                _ => panic!("expected empty tensor"),
            };
            assert_eq!(shape, vec![2, 1, 0]);
            let Value::Tensor(indices) = result.indices else {
                panic!("expected empty indices")
            };
            assert_eq!(indices.shape, vec![2, 1, 0]);
        }
    }
}

#[tokio::test]
async fn topk_output_and_argument_preflight_precedes_provider_lookup() {
    let providerless = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
        shape: vec![2, 1],
        device_id: u32::MAX - 393,
        buffer_id: u64::MAX - 393,
        descriptor: Default::default(),
    });
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    for kind in [TopKKind::Max, TopKKind::Min] {
        {
            let _outputs = crate::output_count::push_output_count(Some(3));
            let error = evaluate_topk(kind, providerless.clone(), &[Value::Num(1.)])
                .await
                .unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:topk:InvalidArgument"));
            assert!(error.message().contains("maximum is 2"));
        }
        let error = evaluate_topk(kind, providerless.clone(), &[Value::Num(-1.)])
            .await
            .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:topk:InvalidArgument"));
        let _discarded = crate::output_count::push_output_count(Some(0));
        assert!(
            evaluate_topk(kind, providerless.clone(), &[Value::Num(-1.)])
                .await
                .is_err()
        );
    }
}

#[tokio::test]
async fn maxk_defaults_to_first_nonsingleton_dimension() {
    let input = tensor(vec![1.0, 4.0, 3.0, 2.0, 6.0, 5.0], vec![3, 2]);
    let eval = evaluate_topk(TopKKind::Max, input, &[Value::Num(2.0)])
        .await
        .unwrap();
    let Value::Tensor(values) = eval.values else {
        panic!("expected tensor");
    };
    assert_eq!(values.shape, vec![2, 2]);
    assert_eq!(self::values(&values), vec![4.0, 3.0, 6.0, 5.0]);
    let Value::Tensor(indices) = eval.indices else {
        panic!("expected indices");
    };
    assert_eq!(self::values(&indices), vec![2.0, 3.0, 2.0, 3.0]);
}

#[tokio::test]
async fn mink_supports_explicit_row_dimension() {
    let input = tensor(vec![3.0, 4.0, 1.0, 2.0, 5.0, 6.0], vec![2, 3]);
    let eval = evaluate_topk(TopKKind::Min, input, &[Value::Num(2.0), Value::Num(2.0)])
        .await
        .unwrap();
    let Value::Tensor(values) = eval.values else {
        panic!("expected tensor");
    };
    assert_eq!(values.shape, vec![2, 2]);
    assert_eq!(self::values(&values), vec![1.0, 2.0, 3.0, 4.0]);
    let Value::Tensor(indices) = eval.indices else {
        panic!("expected indices");
    };
    assert_eq!(self::values(&indices), vec![2.0, 2.0, 1.0, 1.0]);
}

#[tokio::test]
async fn maxk_clamps_k_to_dimension_length() {
    let input = tensor(vec![2.0, 1.0, 3.0], vec![3, 1]);
    let eval = evaluate_topk(TopKKind::Max, input, &[Value::Num(10.0)])
        .await
        .unwrap();
    let Value::Tensor(values) = eval.values else {
        panic!("expected tensor");
    };
    assert_eq!(values.shape, vec![3, 1]);
    assert_eq!(self::values(&values), vec![3.0, 2.0, 1.0]);
}

#[tokio::test]
async fn topk_dimension_greater_than_rank_returns_input_and_one_indices() {
    let input = tensor(vec![2.0, 1.0, 3.0], vec![3, 1]);
    let eval = evaluate_topk(TopKKind::Max, input, &[Value::Num(2.0), Value::Num(5.0)])
        .await
        .unwrap();
    let Value::Tensor(values) = eval.values else {
        panic!("expected tensor");
    };
    assert_eq!(values.shape, vec![3, 1]);
    assert_eq!(self::values(&values), vec![2.0, 1.0, 3.0]);
    let Value::Tensor(indices) = eval.indices else {
        panic!("expected indices");
    };
    assert_eq!(self::values(&indices), vec![1.0, 1.0, 1.0]);
}

#[tokio::test]
async fn topk_rejects_invalid_k() {
    let input = tensor(vec![1.0, 2.0], vec![2, 1]);
    let err = evaluate_topk(TopKKind::Max, input, &[Value::Num(-1.0)])
        .await
        .unwrap_err();
    assert!(err.message().contains("k must be nonnegative"));
}

#[tokio::test]
async fn topk_allows_zero_k() {
    let input = tensor(vec![1.0, 2.0], vec![2, 1]);
    let eval = evaluate_topk(TopKKind::Max, input, &[Value::Num(0.0)])
        .await
        .unwrap();
    let Value::Tensor(values) = eval.values else {
        panic!("expected tensor");
    };
    assert_eq!(values.shape, vec![0, 1]);
    assert!(values.is_empty());
}

#[tokio::test]
async fn builtin_wraps_multiple_outputs() {
    let input = tensor(vec![1.0, 3.0, 2.0], vec![1, 3]);
    let _guard = crate::output_count::push_output_count(Some(2));
    let result = maxk_builtin(input, vec![Value::Num(2.0)]).await.unwrap();
    let values = outputs(result);
    assert_eq!(values.len(), 2);
    let Value::Tensor(selected) = &values[0] else {
        panic!("expected tensor");
    };
    assert_eq!(self::values(selected), vec![3.0, 2.0]);
    let Value::Tensor(indices) = &values[1] else {
        panic!("expected indices");
    };
    assert_eq!(self::values(indices), vec![2.0, 3.0]);
}

#[tokio::test]
async fn topk_rejects_more_than_two_outputs() {
    let _guard = crate::output_count::push_output_count(Some(3));
    let err = maxk_builtin(
        tensor(vec![1.0, 3.0, 2.0], vec![1, 3]),
        vec![Value::Num(2.0)],
    )
    .await
    .expect_err("too many outputs");
    assert!(err.message().contains("maximum is 2"));
}
