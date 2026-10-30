use super::*;

#[test]
fn rmmissing_removes_rows_and_columns() {
    let value = tensor(vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0], vec![3, 2]);
    let result = block_on(rmmissing_builtin(None, value, Vec::new())).unwrap();
    assert!(
        matches!(result, Value::Tensor(t) if t.shape() == [2, 2] && t.materialize_f64() == vec![1.0, 2.0, 4.0, 5.0])
    );

    let value = tensor(vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0], vec![3, 2]);
    let result = block_on(rmmissing_builtin(None, value, vec![Value::Num(2.0)])).unwrap();
    assert!(
        matches!(result, Value::Tensor(t) if t.shape() == [3, 1] && t.materialize_f64() == vec![4.0, 5.0, 6.0])
    );
}

#[test]
fn rmmissing_rejects_invalid_options_before_resident_download() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![2, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let error = block_on(rmmissing_builtin(
        None,
        Value::GpuTensor(allocation.handle.clone()),
        vec![Value::from("not-a-removal-option")],
    ))
    .expect_err("invalid options must be admitted before provider work");
    assert_eq!(error.identifier(), Some("RunMat:missing:InvalidArgument"));
    assert_eq!(allocation.probe.retirements(), 0);
    allocation.handle.release().unwrap();
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn rmmissing_rejects_more_than_two_outputs_before_resident_download() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![2, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let error = block_on(rmmissing_builtin(
        None,
        Value::GpuTensor(allocation.handle.clone()),
        Vec::new(),
    ))
    .expect_err("rmmissing has at most two outputs");
    assert_eq!(error.identifier(), Some("RunMat:missing:InvalidArgument"));
    assert_eq!(allocation.probe.retirements(), 0);
    allocation.handle.release().unwrap();
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn rmmissing_typed_integer_tensor_preserves_storage_and_reports_no_missing() {
    let expected = IntegerStorage::U64(vec![1, u64::MAX, 3, 4]);
    let input = Tensor::new_integer(expected.clone(), vec![2, 2]).expect("integer tensor");

    let options = RemoveOptions::parse(crate::call::FunctionArgumentView::from_slice(&[
        Value::from("rows"),
    ]))
    .unwrap();
    let result = remove_missing_tensor(input, options).unwrap();

    match result {
        (Value::Tensor(tensor), mask) => {
            assert_eq!(tensor.integer_storage(), Some(&expected));
            assert_eq!(mask.data, vec![0, 0]);
            assert_eq!(mask.shape, vec![2, 1]);
        }
        other => panic!("expected tensor and mask, got {other:?}"),
    }
}

#[test]
fn rmmissing_typed_integer_vector_mask_uses_storage_len_not_mirror() {
    let expected = IntegerStorage::I16(vec![1, 2, 3]);
    let input = Tensor::new_integer(expected.clone(), vec![1, 3]).expect("integer tensor");

    let options = RemoveOptions::parse(crate::call::FunctionArgumentView::from_slice(&[
        Value::from("rows"),
    ]))
    .unwrap();
    let result = remove_missing_tensor(input, options).unwrap();

    match result {
        (Value::Tensor(tensor), mask) => {
            assert_eq!(tensor.integer_storage(), Some(&expected));
            assert_eq!(mask.data, vec![0, 0, 0]);
            assert_eq!(mask.shape, vec![1, 3]);
        }
        other => panic!("expected tensor and mask, got {other:?}"),
    }
}

#[test]
fn rmmissing_resident_integer_restores_value_and_mask_through_exact_owner() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new_integer(
            IntegerStorage::U64(vec![1, 9_007_199_254_740_993]),
            vec![1, 2],
        )
        .expect("integer tensor");
        let source = gpu_helpers::upload_tensor(provider, &input).expect("upload integer");
        let source = source
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .unwrap();
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let result = block_on(rmmissing_builtin(
            None,
            Value::GpuTensor(source.clone()),
            Vec::new(),
        ))
        .expect("resident rmmissing");
        let Value::OutputList(outputs) = result else {
            panic!("expected output list");
        };
        assert_eq!(outputs.len(), 2);
        for output in &outputs {
            assert!(
                matches!(output, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_explicit(handle))
            );
        }
        assert!(
            matches!(&outputs[1], Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
        );
        let gathered_value = test_support::gather(outputs[0].clone()).expect("gather value");
        assert_eq!(
            gathered_value.integer_storage(),
            Some(&IntegerStorage::U64(vec![1, 9_007_199_254_740_993]))
        );
        let gathered_mask = test_support::gather(outputs[1].clone()).expect("gather mask");
        assert_eq!(gathered_mask.shape(), &[1, 2]);
        assert_eq!(gathered_mask.materialize_f64(), vec![0.0, 0.0]);
        assert!(gpu_helpers::exact_provider_for_handle(&source).is_some());
        for output in outputs.try_into_owned().expect("unique missing outputs") {
            if let Value::GpuTensor(handle) = output {
                handle.release().unwrap();
            }
        }
        source.release().unwrap();
    });
}

#[test]
fn rmmissing_cell_arrays_use_cell_row_major_order() {
    let value = Value::Cell(
        CellArray::new(
            vec![
                Value::Num(1.0),
                Value::StringArray(
                    StringArray::from_elements_unbounded(vec![MISSING_TEXT], vec![1, 1]).unwrap(),
                ),
                Value::Num(2.0),
                Value::Num(3.0),
            ],
            2,
            2,
        )
        .unwrap(),
    );
    let result = block_on(rmmissing_builtin(None, value.clone(), Vec::new())).unwrap();
    match result {
        Value::Cell(cell) => {
            assert_eq!((cell.rows, cell.cols), (1, 2));
            assert_eq!(cell.get(0, 0).unwrap(), Value::Num(2.0));
            assert_eq!(cell.get(0, 1).unwrap(), Value::Num(3.0));
        }
        other => panic!("expected cell result, got {other:?}"),
    }

    let result = block_on(rmmissing_builtin(None, value, vec![Value::Num(2.0)])).unwrap();
    match result {
        Value::Cell(cell) => {
            assert_eq!((cell.rows, cell.cols), (2, 1));
            assert_eq!(cell.get(0, 0).unwrap(), Value::Num(1.0));
            assert_eq!(cell.get(1, 0).unwrap(), Value::Num(2.0));
        }
        other => panic!("expected cell result, got {other:?}"),
    }
}
