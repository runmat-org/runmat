use super::*;
#[test]
fn ttest2_positive_dimension_above_old_cap_and_unrepresentable_metadata() {
    let _guard = crate::output_count::push_output_count(Some(4)).unwrap();
    let outputs = output_list(
        block_on(ttest2_builtin(
            None,
            Value::Num(1.0),
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("Dim"),
                Value::Num(1025.0),
            ]),
        ))
        .unwrap(),
    );
    let Value::Tensor(ci) = &outputs[2] else {
        panic!("ci")
    };
    assert_eq!(ci.shape().len(), 1025);
    assert_eq!(ci.shape()[0], 2);
    let x = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    assert!(evaluate(
        &x,
        &x,
        Options {
            dim: Some(usize::MAX - 1),
            ..Default::default()
        }
    )
    .is_err());
}
#[test]
fn ttest2_equal_variance_vector_outputs_match_reference_values() {
    let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap());
    let y = Value::Tensor(Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![4, 1]).unwrap());
    let _guard = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
    let outputs = output_list(
        block_on(ttest2_builtin(
            None,
            x,
            y,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap(),
    );
    assert_eq!(outputs[0], Value::Bool(false));
    assert_close(
        match outputs[1] {
            Value::Num(p) => p,
            ref other => panic!("expected p, got {other:?}"),
        },
        0.133_974_596,
        1.0e-8,
    );
    match &outputs[2] {
        Value::Tensor(ci) => {
            assert_eq!(ci.shape(), &[2, 1]);
            assert_close(ci.materialize_f64()[0], -6.031_813, 1.0e-6);
            assert_close(ci.materialize_f64()[1], 1.031_813, 1.0e-6);
        }
        other => panic!("expected ci tensor, got {other:?}"),
    }
    match &outputs[3] {
        Value::Struct(stats) => {
            assert_close(
                match stats.fields.get("tstat").unwrap() {
                    Value::Num(v) => *v,
                    other => panic!("tstat {other:?}"),
                },
                -1.732_051,
                1.0e-6,
            );
            assert_eq!(stats.fields.get("df").unwrap(), &Value::Num(6.0));
            assert_close(
                match stats.fields.get("sd").unwrap() {
                    Value::Num(v) => *v,
                    other => panic!("sd {other:?}"),
                },
                2.041_241,
                1.0e-6,
            );
        }
        other => panic!("expected stats struct, got {other:?}"),
    }
}

#[test]
fn ttest2_unequal_variance_tail_and_nan_omission() {
    let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, f64::NAN, 4.0], vec![5, 1]).unwrap());
    let y = Value::Tensor(Tensor::new(vec![10.0, 11.0, 12.0, 13.0], vec![4, 1]).unwrap());
    let _guard = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
    let outputs = output_list(
        block_on(ttest2_builtin(
            None,
            x,
            y,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("Vartype"),
                Value::from("unequal"),
                Value::from("Tail"),
                Value::from("left"),
                Value::from("Alpha"),
                Value::Num(0.01),
            ]),
        ))
        .unwrap(),
    );
    assert_eq!(outputs[0], Value::Bool(true));
    match outputs[1] {
        Value::Num(p) => assert!(p < 0.001),
        ref other => panic!("expected p, got {other:?}"),
    }
    match &outputs[2] {
        Value::Tensor(ci) => {
            assert_eq!(ci.materialize_f64()[0], f64::NEG_INFINITY);
            assert!(ci.materialize_f64()[1] < 0.0);
        }
        other => panic!("expected ci tensor, got {other:?}"),
    }
    match &outputs[3] {
        Value::Struct(stats) => match stats.fields.get("sd").unwrap() {
            Value::Tensor(sd) => {
                assert_eq!(sd.shape(), &[2, 1]);
                assert_close(sd.materialize_f64()[0], 1.290_994, 1.0e-6);
                assert_close(sd.materialize_f64()[1], 1.290_994, 1.0e-6);
            }
            other => panic!("expected unequal sd tensor, got {other:?}"),
        },
        other => panic!("expected stats struct, got {other:?}"),
    }
}

#[test]
fn ttest2_matrix_dim_outputs_shape_and_invalid_samples() {
    let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0], vec![3, 2]).unwrap());
    let y = Value::Tensor(Tensor::new(vec![1.0, 1.0, 1.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap());
    {
        let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let outputs = output_list(
            block_on(ttest2_builtin(
                None,
                x,
                y,
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .unwrap(),
        );
        match &outputs[0] {
            Value::LogicalArray(h) => assert_eq!(h.shape, vec![1, 2]),
            other => panic!("expected h logical array, got {other:?}"),
        }
        match &outputs[1] {
            Value::Tensor(p) => assert_eq!(p.shape(), &[1, 2]),
            other => panic!("expected p tensor, got {other:?}"),
        }
    }

    let row_x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 5.0, 7.0], vec![2, 2]).unwrap());
    let row_y = Value::Tensor(Tensor::new(vec![2.0, 2.0, 4.0, 4.0], vec![2, 2]).unwrap());
    let out = block_on(ttest2_builtin(
        None,
        row_x,
        row_y,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Dim"),
            Value::Num(2.0),
        ]),
    ))
    .unwrap();
    match out {
        Value::LogicalArray(h) => assert_eq!(h.shape, vec![2, 1]),
        other => panic!("expected dim-2 h, got {other:?}"),
    }
}

#[test]
fn ttest2_rejects_bad_options_and_shapes() {
    let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
    let y = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
    let err = block_on(ttest2_builtin(
        None,
        x,
        y,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(err.message.contains("same size"));

    let err = block_on(ttest2_builtin(
        None,
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Tail"),
            Value::from("sideways"),
        ]),
    ))
    .unwrap_err();
    assert!(err.message.contains("Tail"));
}

#[test]
fn ttest2_host_invalid_controls_precede_sample_provider_access() {
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![4, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("genuine unregistered resident fixture")
            .handle,
    );
    let y = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap());
    let error = block_on(ttest2_builtin(
        None,
        resident,
        y,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Tail"),
            Value::from("sideways"),
        ]),
    ))
    .expect_err("host-invalid Tail must be rejected before sample gather");
    assert!(error.message.contains("unsupported Tail"), "{error}");
    assert_eq!(
        error.identifier.as_deref(),
        Some("RunMat:ttest2:InvalidArgument")
    );
}

#[test]
fn ttest2_sample_provider_failure_preserves_nested_cause() {
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![4, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("genuine unregistered resident fixture")
            .handle,
    );
    let y = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap());
    let error = block_on(ttest2_builtin(
        None,
        resident,
        y,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("unregistered resident sample must fail at provider access");
    assert_eq!(
        error.identifier.as_deref(),
        Some("RunMat:ttest2:InvalidArgument")
    );
    assert!(
        error.source.is_some(),
        "provider failure must remain a nested cause"
    );
}

#[test]
fn ttest2_rejects_too_many_outputs_and_huge_dim() {
    let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap());
    let y = Value::Tensor(Tensor::new(vec![1.0, 2.0, 4.0], vec![3, 1]).unwrap());
    let _guard = crate::output_count::push_output_count(Some(5)).expect("output-count scope");
    let err = block_on(ttest2_builtin(
        None,
        x,
        y,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(err.message.contains("too many output arguments"));

    let err = block_on(ttest2_builtin(
        None,
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Dim"),
            Value::Num(1.0e20),
        ]),
    ))
    .unwrap_err();
    assert!(err.message.contains("Dim"));
}

#[test]
fn ttest2_numeric_options_read_typed_integer_storage_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap());
    let y = Value::Tensor(Tensor::new(vec![1.0, 2.0, 4.0, 8.0], vec![4, 1]).unwrap());
    let out = block_on(ttest2_builtin(
        None,
        x,
        y,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Dim"),
            poisoned_integer_scalar(IntegerStorage::U64(vec![1])),
        ]),
    ))
    .unwrap();
    assert_eq!(out, Value::Bool(false));

    let x = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
    let y = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
    let err = block_on(ttest2_builtin(
        None,
        x,
        y,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Alpha"),
            poisoned_integer_scalar(IntegerStorage::I64(vec![1])),
        ]),
    ))
    .unwrap_err();
    assert!(err.message.contains("Alpha"));
}

#[test]
fn ttest2_sample_values_read_typed_integer_storage_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = poisoned_integer_tensor(IntegerStorage::I16(vec![1, 2, 3, 4]), vec![4, 1]);
    let y = poisoned_integer_tensor(IntegerStorage::U16(vec![1, 2, 4, 8]), vec![4, 1]);
    let out = block_on(ttest2_builtin(
        None,
        x,
        y,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    assert_eq!(out, Value::Bool(false));
}

#[test]
fn ttest2_gates_integer_roles_and_rejects_lossy_boundaries() {
    let integer = poisoned_integer_tensor(IntegerStorage::U8(vec![1]), vec![1, 1]);
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = ensure_ttest2_extensions(&integer, &Value::Num(1.0), &[])
        .expect_err("integer sample must gate");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:Ttest2IntegerSampleExtension")
    );
    let error = ensure_ttest2_extensions(
        &Value::Num(1.0),
        &Value::Num(1.0),
        &[Value::from("Dim"), integer],
    )
    .expect_err("integer option must gate");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:Ttest2IntegerOptionExtension")
    );
    drop(_strict);

    let wide = Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1])
        .expect("wide integer");
    let error = ensure_exact_ttest2_integer_tensor(&wide, "sample x")
        .expect_err("lossy sample must reject");
    assert!(error.message().contains("exactly representable"));
}

#[test]
fn ttest2_empty_and_nd_outputs_preserve_shapes() {
    let empty_x = Value::Tensor(Tensor::new(Vec::new(), vec![3, 0]).unwrap());
    let empty_y = Value::Tensor(Tensor::new(Vec::new(), vec![3, 0]).unwrap());
    let _guard = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
    let outputs = output_list(
        block_on(ttest2_builtin(
            None,
            empty_x,
            empty_y,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap(),
    );
    match &outputs[0] {
        Value::LogicalArray(h) => assert_eq!(h.shape, vec![1, 0]),
        other => panic!("expected empty h logical array, got {other:?}"),
    }
    match &outputs[1] {
        Value::Tensor(p) => assert_eq!(p.shape(), &[1, 0]),
        other => panic!("expected empty p tensor, got {other:?}"),
    }
    match &outputs[2] {
        Value::Tensor(ci) => assert_eq!(ci.shape(), &[2, 0]),
        other => panic!("expected empty ci tensor, got {other:?}"),
    }

    drop(_guard);
    let data = (1..=8).map(|value| value as f64).collect::<Vec<_>>();
    let x = Value::Tensor(Tensor::new(data.clone(), vec![2, 1, 4]).unwrap());
    let y = Value::Tensor(
        Tensor::new(
            data.iter().map(|value| value + 1.0).collect(),
            vec![2, 1, 4],
        )
        .unwrap(),
    );
    let _guard = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
    let outputs = output_list(
        block_on(ttest2_builtin(
            None,
            x,
            y,
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap(),
    );
    match &outputs[2] {
        Value::Tensor(ci) => assert_eq!(ci.shape(), &[2, 1, 4]),
        other => panic!("expected nd ci tensor, got {other:?}"),
    }
    match &outputs[3] {
        Value::Struct(stats) => match stats.fields.get("sd").unwrap() {
            Value::Tensor(sd) => assert_eq!(sd.shape(), &[1, 1, 4]),
            other => panic!("expected nd sd tensor, got {other:?}"),
        },
        other => panic!("expected stats struct, got {other:?}"),
    }
}
