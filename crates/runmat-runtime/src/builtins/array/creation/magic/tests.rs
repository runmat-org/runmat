use super::*;

#[test]
fn automatic_complex_control_materializes_without_losing_its_real_component() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let input = runmat_value::ComplexTensor::new(vec![(3.9, 8.0)], vec![1, 1]).unwrap();
        let handle =
            crate::builtins::common::gpu_helpers::upload_complex_tensor(provider, &input).unwrap();
        let handle = handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .unwrap();
        let result = magic_builtin(vec![Value::GpuTensor(handle)]).unwrap();
        let Value::Tensor(result) = result else {
            panic!("expected host square");
        };
        assert_eq!(result.shape(), vec![3, 3]);
        assert_magic_properties(&result.materialize_f64(), 3);
    });
}

#[test]
fn explicit_device_control_rejects_without_provider_access() {
    let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("genuine unregistered allocation")
        .handle;
    let handle = handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .unwrap();
    let error = magic_builtin(vec![Value::GpuTensor(handle)]).unwrap_err();
    assert!(error.message().contains("gpuArray input is not supported"));
}

#[test]
fn requested_outputs_reject_before_order_evaluation() {
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let error = magic_builtin(vec![Value::Num(f64::INFINITY)]).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:magic:TooManyOutputs"));
}

#[test]
fn discarded_output_preserves_validation() {
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    assert!(magic_builtin(vec![Value::Num(3.0)]).is_ok());
    assert!(magic_builtin(vec![Value::Num(f64::INFINITY)]).is_err());
}

#[test]
fn invalid_orders_fail_without_allocation() {
    for value in [f64::NAN, f64::INFINITY, -1.0] {
        assert!(magic_builtin(vec![Value::Num(value)]).is_err());
    }
    assert!(magic_tensor(usize::MAX).is_err());
}
use futures::executor::block_on;
use runmat_value::ComplexTensor;
use runmat_value::IntegerStorage;

fn magic_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    block_on(super::magic_builtin(
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
}

#[test]
fn magic_two_matches_documented_degenerate_square() {
    let value = magic_builtin(vec![Value::Num(2.0)]).expect("magic(2)");
    let Value::Tensor(tensor) = value else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.shape(), vec![2, 2]);
    assert_eq!(tensor.materialize_f64(), vec![1.0, 4.0, 3.0, 2.0]);
}

#[test]
fn magic_uses_floor_of_real_first_element() {
    let input = Tensor::new(vec![3.9, 99.0], vec![1, 2]).expect("input");
    let value = magic_builtin(vec![Value::Tensor(input)]).expect("magic");
    let Value::Tensor(tensor) = value else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.shape(), vec![3, 3]);

    let complex =
        ComplexTensor::new(vec![(4.8, 7.0), (12.0, 0.0)], vec![1, 2]).expect("complex input");
    let value = magic_builtin(vec![Value::ComplexTensor(complex)]).expect("magic");
    let Value::Tensor(tensor) = value else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.shape(), vec![4, 4]);

    let logical = runmat_value::LogicalArray::new(vec![1, 0], vec![1, 2]).expect("logical input");
    assert_eq!(
        block_on(parse_order_value(&Value::LogicalArray(logical))).unwrap(),
        1
    );
    assert_eq!(
        block_on(parse_order_value(&Value::CharArray(
            runmat_value::CharArray::new_row("\u{3}\u{7}"),
        )))
        .unwrap(),
        3
    );
}

#[test]
fn magic_zero_is_empty() {
    let value = magic_builtin(vec![Value::Num(0.0)]).expect("magic");
    match value {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), vec![0, 0]);
            assert!(tensor.materialize_f64().is_empty());
        }
        other => panic!("expected empty tensor, got {other:?}"),
    }
}

#[test]
fn magic_parses_all_integer_classes_without_f64_rounding() {
    let values = [
        IntValue::I8(3),
        IntValue::I16(3),
        IntValue::I32(3),
        IntValue::I64(3),
        IntValue::U8(3),
        IntValue::U16(3),
        IntValue::U32(3),
        IntValue::U64(3),
    ];
    for value in values {
        assert!(matches!(
            block_on(parse_order_value(&Value::Int(value))),
            Ok(3)
        ));
    }

    let exact = 9_007_199_254_740_993_u64;
    let tensor =
        Tensor::new_integer(IntegerStorage::U64(vec![exact]), vec![1, 1]).expect("typed scalar");
    assert!(matches!(
        block_on(parse_order_value(&Value::Tensor(tensor))),
        Ok(value) if value == exact as usize
    ));
}

#[test]
fn magic_three_matches_matlab() {
    let value = magic_builtin(vec![Value::Num(3.0)]).expect("magic");
    let tensor = match value {
        Value::Tensor(tensor) => tensor,
        other => panic!("expected tensor, got {other:?}"),
    };
    assert_eq!(tensor.shape(), vec![3, 3]);
    let expected = vec![8.0, 3.0, 4.0, 1.0, 5.0, 9.0, 6.0, 7.0, 2.0];
    assert_eq!(tensor.materialize_f64(), expected);
}

#[test]
fn magic_four_matches_matlab() {
    let value = magic_builtin(vec![Value::Num(4.0)]).expect("magic");
    let tensor = match value {
        Value::Tensor(tensor) => tensor,
        other => panic!("expected tensor, got {other:?}"),
    };
    assert_eq!(tensor.shape(), vec![4, 4]);
    let expected = vec![
        16.0, 5.0, 9.0, 4.0, 2.0, 11.0, 7.0, 14.0, 3.0, 10.0, 6.0, 15.0, 13.0, 8.0, 12.0, 1.0,
    ];
    assert_eq!(tensor.materialize_f64(), expected);
}

/// Verify that every value 1..=n² appears exactly once and that all rows,
/// columns, and both diagonals sum to the magic constant n*(n²+1)/2.
fn assert_magic_properties(data: &[f64], n: usize) {
    let n_sq = n * n;
    let magic_const = n * (n_sq + 1) / 2;

    // All values 1..=n² present
    let mut sorted: Vec<u64> = data.iter().map(|&v| v as u64).collect();
    sorted.sort();
    let expected: Vec<u64> = (1..=n_sq as u64).collect();
    assert_eq!(
        sorted, expected,
        "magic({n}): not a permutation of 1..={n_sq}"
    );

    // Row sums (column-major storage: element (r,c) = data[r + c*n])
    for r in 0..n {
        let row_sum: f64 = (0..n).map(|c| data[r + c * n]).sum();
        assert_eq!(
            row_sum as usize, magic_const,
            "magic({n}): row {r} sum mismatch"
        );
    }
    // Column sums
    for c in 0..n {
        let col_sum: f64 = (0..n).map(|r| data[r + c * n]).sum();
        assert_eq!(
            col_sum as usize, magic_const,
            "magic({n}): col {c} sum mismatch"
        );
    }
    // Main diagonal
    let diag1: f64 = (0..n).map(|i| data[i + i * n]).sum();
    assert_eq!(
        diag1 as usize, magic_const,
        "magic({n}): main diagonal sum mismatch"
    );
    // Anti-diagonal
    let diag2: f64 = (0..n).map(|i| data[i + (n - 1 - i) * n]).sum();
    assert_eq!(
        diag2 as usize, magic_const,
        "magic({n}): anti-diagonal sum mismatch"
    );
}

#[test]
fn magic_six_matches_matlab() {
    let value = magic_builtin(vec![Value::Num(6.0)]).expect("magic");
    let tensor = match value {
        Value::Tensor(tensor) => tensor,
        other => panic!("expected tensor, got {other:?}"),
    };
    assert_eq!(tensor.shape(), vec![6, 6]);
    let expected = vec![
        35.0, 3.0, 31.0, 8.0, 30.0, 4.0, 1.0, 32.0, 9.0, 28.0, 5.0, 36.0, 6.0, 7.0, 2.0, 33.0,
        34.0, 29.0, 26.0, 21.0, 22.0, 17.0, 12.0, 13.0, 19.0, 23.0, 27.0, 10.0, 14.0, 18.0, 24.0,
        25.0, 20.0, 15.0, 16.0, 11.0,
    ];
    assert_eq!(tensor.materialize_f64(), expected);
}

#[test]
fn magic_ten_is_valid_magic_square() {
    let value = magic_builtin(vec![Value::Num(10.0)]).expect("magic");
    let tensor = match value {
        Value::Tensor(tensor) => tensor,
        other => panic!("expected tensor, got {other:?}"),
    };
    assert_eq!(tensor.shape(), vec![10, 10]);
    assert_magic_properties(&tensor.materialize_f64(), 10);
}

#[test]
fn magic_fourteen_is_valid_magic_square() {
    let value = magic_builtin(vec![Value::Num(14.0)]).expect("magic");
    let tensor = match value {
        Value::Tensor(tensor) => tensor,
        other => panic!("expected tensor, got {other:?}"),
    };
    assert_eq!(tensor.shape(), vec![14, 14]);
    assert_magic_properties(&tensor.materialize_f64(), 14);
}
