use super::super::entrypoints::fillmissing_builtin;
use super::*;

#[test]
fn logical_nd_input_preserves_carrier_and_returns_shape_matched_mask() {
    use runmat_types::{
        CallRequest, LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount,
    };
    for (data, shape) in [(vec![1, 0, 1, 0], vec![2, 1, 2]), (vec![], vec![2, 0, 3])] {
        let input = Value::LogicalArray(LogicalArray::new(data, shape.clone()).unwrap());
        let method = Value::from("previous");
        let request = CallRequest {
            arguments: [&input, &method]
                .into_iter()
                .map(crate::value_fact::value_fact)
                .collect(),
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("previous".into()),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        };
        let inferred = runmat_builtins::infer_catalog_call(
            &runmat_builtins::FILLMISSING_CATALOG_ENTRY,
            &request,
        );
        let options =
            FillOptions::parse(crate::call::FunctionArgumentView::from_slice(&[method])).unwrap();
        let (output, mask) =
            super::super::filling::fill_missing_value(None, input.clone(), &options).unwrap();
        assert_eq!(output, input);
        assert_eq!(mask.shape, shape);
        assert!(mask.data.iter().all(|value| *value == 0));
        for (expected, actual) in inferred
            .outputs
            .iter()
            .zip([output, Value::LogicalArray(mask)])
        {
            let live = crate::value_fact::value_fact(&actual);
            assert_eq!(expected.kind, live.kind);
            assert_eq!(expected.storage, live.storage);
            assert!(expected.shape.is_proven_equivalent(&live.shape));
        }
    }
}

#[test]
fn empty_nd_string_input_preserves_shape_and_mask_geometry() {
    let shape = vec![2, 0, 3];
    let input = Value::StringArray(StringArray::new(vec![], shape.clone()).unwrap());
    let options = FillOptions::parse(crate::call::FunctionArgumentView::from_slice(&[
        Value::from("previous"),
    ]))
    .unwrap();
    let (output, mask) =
        super::super::filling::fill_missing_value(None, input.clone(), &options).unwrap();
    assert_eq!(output, input);
    assert_eq!(mask.shape, shape);
    assert!(mask.is_empty());
}

#[test]
fn scalar_mask_catalog_storage_matches_live_runtime_fact() {
    use runmat_types::{
        CallRequest, LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount,
    };
    let input = Value::Num(f64::NAN);
    let method = Value::from("constant");
    let replacement = Value::Num(5.0);
    let request = CallRequest {
        arguments: [&input, &method, &replacement]
            .into_iter()
            .map(crate::value_fact::value_fact)
            .collect(),
        literals: LiteralContext::new(vec![
            LiteralValue::Unknown,
            LiteralValue::String("constant".into()),
            LiteralValue::Number(5.0),
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
    };
    let inferred =
        runmat_builtins::infer_catalog_call(&runmat_builtins::FILLMISSING_CATALOG_ENTRY, &request);
    let options = FillOptions::parse(crate::call::FunctionArgumentView::from_slice(&[
        method,
        replacement,
    ]))
    .unwrap();
    let (_, mask) = super::super::filling::fill_missing_value(None, input, &options).unwrap();
    let live = crate::value_fact::value_fact(&Value::LogicalArray(mask));
    assert_eq!(inferred.outputs[1].kind, live.kind);
    assert_eq!(inferred.outputs[1].storage, live.storage);
    assert!(inferred.outputs[1].shape.is_proven_equivalent(&live.shape));
}

#[test]
fn fillmissing_supports_constant_previous_and_linear() {
    let value = tensor(vec![1.0, f64::NAN, 3.0], vec![3, 1]);
    let result = block_on(fillmissing_builtin(
        None,
        value,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("linear")]),
    ))
    .unwrap();
    assert!(matches!(result, Value::Tensor(t) if t.materialize_f64() == vec![1.0, 2.0, 3.0]));

    let value = tensor(vec![1.0, f64::NAN, 3.0], vec![1, 3]);
    let result = block_on(fillmissing_builtin(
        None,
        value,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("linear")]),
    ))
    .unwrap();
    assert!(matches!(result, Value::Tensor(t) if t.materialize_f64() == vec![1.0, 2.0, 3.0]));

    let value = tensor(vec![1.0, f64::NAN, f64::NAN], vec![3, 1]);
    let result = block_on(fillmissing_builtin(
        None,
        value,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("constant"),
            Value::Num(9.0),
        ]),
    ))
    .unwrap();
    assert!(matches!(result, Value::Tensor(t) if t.materialize_f64() == vec![1.0, 9.0, 9.0]));
}

#[test]
fn fillmissing_rejects_invalid_method_before_resident_download() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![2, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let error = block_on(fillmissing_builtin(
        None,
        Value::GpuTensor(allocation.handle.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from(
            "not-a-fill-method",
        )]),
    ))
    .expect_err("invalid method must be admitted before provider work");
    assert_eq!(error.identifier(), Some("RunMat:missing:InvalidArgument"));
    assert_eq!(allocation.probe.retirements(), 0);
    allocation.handle.release().unwrap();
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn fillmissing_scalar_constant_replacement_preserves_selected_value_class() {
    let missing = block_on(fillmissing_builtin(
        None,
        Value::Num(f64::NAN),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("constant"),
            Value::Bool(true),
        ]),
    ))
    .unwrap();
    assert!(matches!(missing, Value::Bool(true)));
    let present = block_on(fillmissing_builtin(
        None,
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("constant"),
            Value::Bool(true),
        ]),
    ))
    .unwrap();
    assert!(matches!(present, Value::Num(2.0)));
}

#[test]
fn fillmissing_rejects_more_than_two_outputs_before_resident_download() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![2, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let error = block_on(fillmissing_builtin(
        None,
        Value::GpuTensor(allocation.handle.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("constant"),
            Value::Num(0.0),
        ]),
    ))
    .expect_err("fillmissing has at most two outputs");
    assert_eq!(error.identifier(), Some("RunMat:missing:InvalidArgument"));
    assert_eq!(allocation.probe.retirements(), 0);
    allocation.handle.release().unwrap();
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn fillmissing_typed_integer_tensor_preserves_storage_and_reports_no_missing() {
    let expected = IntegerStorage::I32(vec![10, 20, 30]);
    let input = Tensor::new_integer(expected.clone(), vec![3, 1]).expect("integer tensor");

    let options = FillOptions::parse(crate::call::FunctionArgumentView::from_slice(&[
        Value::from("constant"),
        Value::Num(0.0),
    ]))
    .unwrap();
    let result = fill_missing_tensor(input, &options).unwrap();

    match result {
        (Value::Tensor(tensor), mask) => {
            assert_eq!(tensor.integer_storage(), Some(&expected));
            assert_eq!(mask.data, vec![0, 0, 0]);
            assert_eq!(mask.shape, vec![3, 1]);
        }
        other => panic!("expected tensor and mask, got {other:?}"),
    }
}

#[test]
fn fillmissing_integer_data_is_gated_and_all_classes_are_exact_noops() {
    let storages = [
        IntegerStorage::I8(vec![-1, 2]),
        IntegerStorage::I16(vec![-1, 2]),
        IntegerStorage::I32(vec![-1, 2]),
        IntegerStorage::I64(vec![-1, 2]),
        IntegerStorage::U8(vec![1, 2]),
        IntegerStorage::U16(vec![1, 2]),
        IntegerStorage::U32(vec![1, 2]),
        IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX]),
    ];
    for storage in storages {
        let input = Value::Tensor(Tensor::new_integer(storage.clone(), vec![2, 1]).unwrap());
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(fillmissing_builtin(
                None,
                input.clone(),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::from("constant"),
                    Value::Num(0.0),
                ]),
            ))
            .expect_err("strict integer fillmissing");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:FillmissingIntegerDataExtension")
            );
        }
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let output = block_on(fillmissing_builtin(
            None,
            input,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("constant"),
                Value::Num(0.0),
            ]),
        ))
        .expect("RunMat integer fillmissing");
        let Value::OutputList(values) = output else {
            panic!("expected output list");
        };
        assert!(
            matches!(&values[0], Value::Tensor(tensor) if tensor.integer_storage() == Some(&storage))
        );
        assert!(
            matches!(&values[1], Value::LogicalArray(mask) if mask.shape == vec![2, 1] && mask.data == vec![0, 0])
        );
    }
}

#[test]
fn fillmissing_text_neighbors_preserve_nd_lines_and_mask_shape() {
    let shape = vec![2, 1, 3];
    for (method, middle) in [
        ("previous", ["a", "b"]),
        ("next", ["c", "d"]),
        ("nearest", ["a", "b"]),
    ] {
        let input = StringArray::from_elements_unbounded(
            vec![
                "a".into(),
                "b".into(),
                MISSING_TEXT.into(),
                MISSING_TEXT.into(),
                "c".into(),
                "d".into(),
            ],
            shape.clone(),
        )
        .unwrap();
        let options = FillOptions::parse(crate::call::FunctionArgumentView::from_slice(&[
            Value::from(method),
            Value::Num(3.0),
        ]))
        .unwrap();
        let (output, mask) = fill_missing_string_array(None, input, &options).unwrap();
        let Value::StringArray(output) = output else {
            panic!("expected string array")
        };
        assert_eq!(
            output.data(),
            ["a", "b", middle[0], middle[1], "c", "d"].map(str::to_owned)
        );
        assert_eq!(output.shape, shape);
        assert_eq!(mask.shape, shape);
        assert_eq!(mask.data, vec![0, 0, 1, 1, 0, 0]);
    }

    use crate::context::host_effect::ManagedHostEffectLedger;
    use crate::context::{
        ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
        RuntimeServicePorts,
    };
    use std::rc::Rc;
    struct ManagedParallel(ManagedHostEffectLedger);
    impl RuntimeParallelService for ManagedParallel {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..RuntimeParallelResources::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }
    let ledger = ManagedHostEffectLedger::new(1 << 20);
    let context = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
        );
    let input = StringArray::from_elements_unbounded(
        vec![
            "a".into(),
            "b".into(),
            MISSING_TEXT.into(),
            MISSING_TEXT.into(),
            "c".into(),
            "d".into(),
        ],
        shape.clone(),
    )
    .unwrap();
    let options = FillOptions::parse(crate::call::FunctionArgumentView::from_slice(&[
        Value::from("nearest"),
        Value::Num(3.0),
    ]))
    .unwrap();
    let (output, mask) = fill_missing_string_array(Some(&context), input, &options).unwrap();
    let Value::StringArray(output) = output else {
        panic!("expected string array")
    };
    assert_eq!(
        output
            .data()
            .iter()
            .map(|value| value.as_str().expect("filled text"))
            .collect::<Vec<_>>(),
        ["a", "b", "a", "b", "c", "d"]
    );
    assert!(output.data().iter().all(|value| value.has_retained_claim()));
    assert_eq!(mask.data, vec![0, 0, 1, 1, 0, 0]);
    assert!(ledger.snapshot().live_reserved_bytes > 0);
    let unique_outer = output.data().as_ptr();
    let indicator = super::super::standardize::indicator_set(&Value::from("a")).unwrap();
    let Value::StringArray(unique) = super::super::standardize::standardize_missing_value(
        Some(&context),
        Value::StringArray(output),
        &indicator,
    )
    .unwrap() else {
        panic!("expected standardized array")
    };
    assert_eq!(unique.data().as_ptr(), unique_outer);
    assert!(unique.data()[0].is_missing());
    let alias = unique.clone();
    let shared_outer = unique.data().as_ptr();
    let indicator = super::super::standardize::indicator_set(&Value::from("b")).unwrap();
    let Value::StringArray(replaced) = super::super::standardize::standardize_missing_value(
        Some(&context),
        Value::StringArray(unique),
        &indicator,
    )
    .unwrap() else {
        panic!("expected shared replacement")
    };
    assert_ne!(replaced.data().as_ptr(), shared_outer);
    assert_eq!(alias.data()[1].as_str(), Some("b"));
    assert!(replaced.data()[1].is_missing());
    drop(replaced);
    drop(alias);
    drop(mask);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn fillmissing_nearest_uses_original_neighbors_and_previous_on_ties() {
    let options = FillOptions::parse(crate::call::FunctionArgumentView::from_slice(&[
        Value::from("nearest"),
    ]))
    .unwrap();
    for shape in [vec![7, 1], vec![1, 7]] {
        let input = Tensor::new(
            vec![f64::NAN, 1.0, f64::NAN, f64::NAN, f64::NAN, 5.0, f64::NAN],
            shape,
        )
        .unwrap();
        let (output, mask) = fill_missing_tensor(input, &options).unwrap();
        let Value::Tensor(output) = output else {
            panic!("expected tensor")
        };
        assert_eq!(
            output.materialize_f64(),
            vec![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0]
        );
        assert_eq!(mask.data, vec![1, 0, 1, 1, 1, 0, 1]);
    }
    let (output, mask) = fill_missing_tensor(
        Tensor::new(vec![f64::NAN; 4], vec![4, 1]).unwrap(),
        &options,
    )
    .unwrap();
    let Value::Tensor(output) = output else {
        panic!("expected tensor")
    };
    assert!(output.materialize_f64().iter().all(|value| value.is_nan()));
    assert_eq!(mask.data, vec![0; 4]);
}

#[test]
fn fillmissing_integer_noop_preserves_nd_and_empty_mask_shapes() {
    let options = FillOptions::parse(crate::call::FunctionArgumentView::from_slice(&[
        Value::from("constant"),
        Value::Num(0.0),
    ]))
    .unwrap();
    for shape in [vec![2, 1, 2], vec![1, 1, 4], vec![2, 0, 3]] {
        let count = shape.iter().product();
        let storage = IntegerStorage::U64(vec![u64::MAX; count]);
        let input = Tensor::new_integer(storage.clone(), shape.clone()).unwrap();
        let (output, mask) = fill_missing_tensor(input, &options).unwrap();
        let Value::Tensor(output) = output else {
            panic!("integer input must retain its tensor carrier");
        };
        assert_eq!(output.integer_storage(), Some(&storage));
        assert_eq!(output.shape(), shape);
        assert_eq!(mask.shape, shape);
        assert_eq!(mask.data, vec![0; count]);
    }
}

#[test]
fn fillmissing_constant_preserves_nd_values_and_mask_shape() {
    let shape = vec![2, 1, 2];
    let input = Tensor::new(vec![1.0, f64::NAN, f64::NAN, 4.0], shape.clone()).unwrap();
    let options = FillOptions::parse(crate::call::FunctionArgumentView::from_slice(&[
        Value::from("constant"),
        Value::Num(7.0),
    ]))
    .unwrap();
    let (output, mask) = fill_missing_tensor(input, &options).unwrap();
    let Value::Tensor(output) = output else {
        panic!("floating input must retain its tensor carrier");
    };
    assert_eq!(output.shape(), shape);
    assert_eq!(output.materialize_f64(), vec![1.0, 7.0, 7.0, 4.0]);
    assert_eq!(mask.shape, shape);
    assert_eq!(mask.data, vec![0, 1, 1, 0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn fillmissing_preserves_rank_zero_and_rank_one_tensor_shapes() {
    for (shape, data, expected) in [
        (Vec::new(), vec![f64::NAN], vec![7.0]),
        (vec![2], vec![f64::NAN, 3.0], vec![7.0, 3.0]),
    ] {
        let result = block_on(fillmissing_builtin(
            None,
            Value::Tensor(Tensor::new(data, shape.clone()).unwrap()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("constant"),
                Value::Num(7.0),
            ]),
        ))
        .expect("fillmissing should preserve full-rank tensor geometry");
        let Value::Tensor(output) = result else {
            panic!("expected tensor output");
        };
        assert_eq!(output.shape(), shape);
        assert_eq!(output.materialize_f64(), expected);
    }
}

#[test]
fn fillmissing_numeric_methods_traverse_each_matrix_and_nd_line_independently() {
    for (method, delta) in [
        ("previous", 0.0),
        ("next", 2.0),
        ("nearest", 0.0),
        ("linear", 1.0),
        ("mean", 1.0),
        ("median", 1.0),
    ] {
        for shape in [vec![3, 2], vec![3, 2, 2]] {
            let groups = shape.iter().product::<usize>() / 3;
            let mut input = Vec::new();
            let mut expected = Vec::new();
            let mut expected_mask = Vec::new();
            for group in 0..groups {
                let base = (group * 10) as f64;
                input.extend([base, f64::NAN, base + 2.0]);
                expected.extend([base, base + delta, base + 2.0]);
                expected_mask.extend([0, 1, 0]);
            }
            let options = FillOptions::parse(crate::call::FunctionArgumentView::from_slice(&[
                Value::from(method),
            ]))
            .unwrap();
            let (output, mask) =
                fill_missing_tensor(Tensor::new(input, shape.clone()).unwrap(), &options).unwrap();
            let Value::Tensor(output) = output else {
                panic!("expected tensor")
            };
            assert_eq!(output.materialize_f64(), expected, "{method}, {shape:?}");
            assert_eq!(output.shape(), shape);
            assert_eq!(mask.shape, shape);
            assert_eq!(mask.data, expected_mask);
        }
        let options = FillOptions::parse(crate::call::FunctionArgumentView::from_slice(&[
            Value::from(method),
            Value::Num(3.0),
        ]))
        .unwrap();
        let shape = vec![2, 2, 3];
        let input = vec![
            1.0,
            10.0,
            20.0,
            30.0,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            3.0,
            12.0,
            22.0,
            32.0,
        ];
        let expected = vec![
            1.0,
            10.0,
            20.0,
            30.0,
            1.0 + delta,
            10.0 + delta,
            20.0 + delta,
            30.0 + delta,
            3.0,
            12.0,
            22.0,
            32.0,
        ];
        let (output, mask) =
            fill_missing_tensor(Tensor::new(input, shape.clone()).unwrap(), &options).unwrap();
        let Value::Tensor(output) = output else {
            panic!("expected tensor")
        };
        assert_eq!(output.materialize_f64(), expected, "{method}, dimension3");
        assert_eq!(output.shape(), shape);
        assert_eq!(mask.shape, shape);
        assert_eq!(mask.data, vec![0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]);
    }
}

#[test]
fn fillmissing_integer_table_variable_and_nested_cell_are_aggregate_gated() {
    let integer =
        Value::Tensor(Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![2, 1]).unwrap());
    let table =
        crate::builtins::table::table_from_columns(vec!["I".to_string()], vec![integer.clone()])
            .unwrap();
    let nested = Value::Cell(
        CellArray::new(
            vec![Value::Cell(CellArray::new(vec![integer], 1, 1).unwrap())],
            1,
            1,
        )
        .unwrap(),
    );
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    for aggregate in [table, nested] {
        let error = block_on(fillmissing_builtin(
            None,
            aggregate,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("constant"),
                Value::Num(0.0),
            ]),
        ))
        .expect_err("strict aggregate integer fillmissing");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:FillmissingAggregateIntegerDataExtension")
        );
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn fillmissing_nested_resident_integer_is_gated_before_provider_access() {
    test_support::with_test_provider(|provider| {
        let handle = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&[u64::MAX]),
                shape: &[1, 1],
            })
            .expect("integer upload");
        let nested = Value::Cell(
            CellArray::new(
                vec![Value::Cell(
                    CellArray::new(vec![Value::GpuTensor(handle.clone())], 1, 1).unwrap(),
                )],
                1,
                1,
            )
            .unwrap(),
        );
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(fillmissing_builtin(
            None,
            nested,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("constant"),
                Value::Num(0.0),
            ]),
        ))
        .expect_err("strict nested resident integer fillmissing");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:FillmissingAggregateIntegerDataExtension")
        );
        handle
            .allocation_lease()
            .validate_live()
            .expect("borrowed input remains live");
        let physical = handle.allocation_lease().downgrade();
        handle.release().expect("release final fixture reference");
        assert!(
            physical.acquire().is_none(),
            "physical lease must expire after final release"
        );
    });
}

#[test]
fn fillmissing_rejects_complex_and_sparse_arrays_instead_of_aggregate_replacement() {
    let complex = Value::ComplexTensor(
        runmat_value::ComplexTensor::new(vec![(f64::NAN, 0.0)], vec![1, 1]).unwrap(),
    );
    let options = FillOptions::parse(crate::call::FunctionArgumentView::from_slice(&[
        Value::from("constant"),
        Value::Num(0.0),
    ]))
    .unwrap();
    let error =
        fill_missing_value(None, complex, &options).expect_err("complex arrays are unsupported");
    assert!(error.message().contains("complex arrays are not supported"));

    let sparse = Value::SparseTensor(
        runmat_value::SparseTensor::new(2, 1, vec![0, 1], vec![0], vec![f64::NAN]).unwrap(),
    );
    let error =
        fill_missing_value(None, sparse, &options).expect_err("sparse arrays are unsupported");
    assert!(error.message().contains("sparse arrays are not supported"));
}

#[test]
fn fillmissing_nearest_uses_original_neighbors() {
    let value = tensor(vec![1.0, f64::NAN, f64::NAN, 4.0], vec![1, 4]);
    let result = block_on(fillmissing_builtin(
        None,
        value,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("nearest")]),
    ))
    .unwrap();
    assert!(matches!(result, Value::Tensor(t) if t.materialize_f64() == vec![1.0, 1.0, 4.0, 4.0]));
}

#[test]
fn fillmissing_mask_marks_only_entries_actually_filled() {
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let value = tensor(vec![f64::NAN, 2.0, 3.0], vec![3, 1]);
    let output = block_on(fillmissing_builtin(
        None,
        value,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("previous")]),
    ))
    .unwrap();
    let Value::OutputList(values) = output else {
        panic!("expected output list");
    };
    assert!(matches!(&values[1], Value::LogicalArray(mask) if mask.data == vec![0, 0, 0]));
}

#[test]
fn fillmissing_rejects_unknown_options() {
    let value = tensor(vec![1.0, f64::NAN], vec![1, 2]);
    let result = block_on(fillmissing_builtin(
        None,
        value,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("constant"),
            Value::Num(0.0),
            Value::from("bogus"),
        ]),
    ));
    assert!(result.is_err());
}
