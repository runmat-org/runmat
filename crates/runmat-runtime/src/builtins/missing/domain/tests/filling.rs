use super::super::entrypoints::fillmissing_builtin;
use super::*;

#[test]
fn fillmissing_supports_constant_previous_and_linear() {
    let value = tensor(vec![1.0, f64::NAN, 3.0], vec![3, 1]);
    let result = block_on(fillmissing_builtin(value, vec![Value::from("linear")])).unwrap();
    assert!(matches!(result, Value::Tensor(t) if t.materialize_f64() == vec![1.0, 2.0, 3.0]));

    let value = tensor(vec![1.0, f64::NAN, 3.0], vec![1, 3]);
    let result = block_on(fillmissing_builtin(value, vec![Value::from("linear")])).unwrap();
    assert!(matches!(result, Value::Tensor(t) if t.materialize_f64() == vec![1.0, 2.0, 3.0]));

    let value = tensor(vec![1.0, f64::NAN, f64::NAN], vec![3, 1]);
    let result = block_on(fillmissing_builtin(
        value,
        vec![Value::from("constant"), Value::Num(9.0)],
    ))
    .unwrap();
    assert!(matches!(result, Value::Tensor(t) if t.materialize_f64() == vec![1.0, 9.0, 9.0]));
}

#[test]
fn fillmissing_typed_integer_tensor_preserves_storage_and_reports_no_missing() {
    let expected = IntegerStorage::I32(vec![10, 20, 30]);
    let input = Tensor::new_integer(expected.clone(), vec![3, 1]).expect("integer tensor");

    let options = FillOptions::parse(&[Value::from("constant"), Value::Num(0.0)]).unwrap();
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
                input.clone(),
                vec![Value::from("constant"), Value::Num(0.0)],
            ))
            .expect_err("strict integer fillmissing");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:FillmissingIntegerDataExtension")
            );
        }
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _outputs = crate::output_count::push_output_count(Some(2));
        let output = block_on(fillmissing_builtin(
            input,
            vec![Value::from("constant"), Value::Num(0.0)],
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
            aggregate,
            vec![Value::from("constant"), Value::Num(0.0)],
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
            nested,
            vec![Value::from("constant"), Value::Num(0.0)],
        ))
        .expect_err("strict nested resident integer fillmissing");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:FillmissingAggregateIntegerDataExtension")
        );
        provider.free(&handle).ok();
    });
}

#[test]
fn fillmissing_rejects_complex_and_sparse_arrays_instead_of_aggregate_replacement() {
    let complex = Value::ComplexTensor(
        runmat_value::ComplexTensor::new(vec![(f64::NAN, 0.0)], vec![1, 1]).unwrap(),
    );
    let options = FillOptions::parse(&[Value::from("constant"), Value::Num(0.0)]).unwrap();
    let error = fill_missing_value(complex, &options).expect_err("complex arrays are unsupported");
    assert!(error.message().contains("complex arrays are not supported"));

    let sparse = Value::SparseTensor(
        runmat_value::SparseTensor::new(2, 1, vec![0, 1], vec![0], vec![f64::NAN]).unwrap(),
    );
    let error = fill_missing_value(sparse, &options).expect_err("sparse arrays are unsupported");
    assert!(error.message().contains("sparse arrays are not supported"));
}

#[test]
fn fillmissing_nearest_uses_original_neighbors() {
    let value = tensor(vec![1.0, f64::NAN, f64::NAN, 4.0], vec![1, 4]);
    let result = block_on(fillmissing_builtin(value, vec![Value::from("nearest")])).unwrap();
    assert!(matches!(result, Value::Tensor(t) if t.materialize_f64() == vec![1.0, 1.0, 4.0, 4.0]));
}

#[test]
fn fillmissing_mask_marks_only_entries_actually_filled() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let value = tensor(vec![f64::NAN, 2.0, 3.0], vec![3, 1]);
    let output = block_on(fillmissing_builtin(value, vec![Value::from("previous")])).unwrap();
    let Value::OutputList(values) = output else {
        panic!("expected output list");
    };
    assert!(matches!(&values[1], Value::LogicalArray(mask) if mask.data == vec![0, 0, 0]));
}

#[test]
fn fillmissing_rejects_unknown_options() {
    let value = tensor(vec![1.0, f64::NAN], vec![1, 2]);
    let result = block_on(fillmissing_builtin(
        value,
        vec![
            Value::from("constant"),
            Value::Num(0.0),
            Value::from("bogus"),
        ],
    ));
    assert!(result.is_err());
}
