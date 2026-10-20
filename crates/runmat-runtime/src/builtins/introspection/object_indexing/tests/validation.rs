use super::*;
#[test]
fn typed_scalar_subscripts_do_not_round_or_saturate() {
    assert_eq!(
        single_subscript_count(&Value::Num(0.0), &Value::Int(IntValue::U64(u64::MAX)), 0, 1)
            .expect("positive uint64 subscript"),
        1
    );
    assert!(
        single_subscript_count(&Value::Num(0.0), &Value::Int(IntValue::I64(-1)), 0, 1).is_err()
    );
}

#[test]
fn num_arguments_integer_metadata_and_double_boundary_are_explicit() {
    assert_eq!(NUM_ARGUMENTS_FROM_SUBSCRIPT_INTEGER_CAPABILITIES.len(), 1);
    assert_eq!(
        NUM_ARGUMENTS_FROM_SUBSCRIPT_INTEGER_CAPABILITIES[0].output_class,
        BuiltinIntegerOutputClassRule::Double
    );
    #[cfg(target_pointer_width = "64")]
    {
        assert_eq!(
            exact_count_as_f64(9_007_199_254_740_992).expect("exact count"),
            9_007_199_254_740_992.0
        );
        let error =
            exact_count_as_f64(9_007_199_254_740_993).expect_err("count is not exact binary64");
        assert_eq!(error.identifier(), NUM_ARGUMENTS_ERROR_COUNT.identifier);
    }
}

#[test]
fn default_brace_rejects_invalid_numeric_subscripts() {
    let target = cell(vec![Value::Num(1.0), Value::Num(2.0)]);
    for selector in [
        Value::Num(0.0),
        Value::Num(f64::NAN),
        Value::Num(1.5),
        Value::Tensor(Tensor::new(vec![1.0, 1.25], vec![1, 2]).expect("indices")),
    ] {
        let descriptor = substruct(crate::OBJECT_INDEX_BRACE, cell(vec![selector]));
        let err = call_err(
            target.clone(),
            descriptor,
            Value::String("Statement".to_string().into()),
        );
        assert_eq!(
            err.identifier(),
            Some("RunMat:numArgumentsFromSubscript:InvalidSubstruct")
        );
    }
}

#[test]
fn default_brace_rejects_unvalidated_subscript_containers() {
    let target = cell(vec![Value::Num(1.0), Value::Num(2.0)]);
    let selector = cell(vec![Value::Num(1.0)]);
    let descriptor = substruct(crate::OBJECT_INDEX_BRACE, cell(vec![selector]));
    let err = call_err(
        target,
        descriptor,
        Value::String("Statement".to_string().into()),
    );
    assert_eq!(
        err.identifier(),
        Some("RunMat:numArgumentsFromSubscript:InvalidSubstruct")
    );
}

#[test]
fn default_brace_shape_overflow_returns_substruct_error() {
    // Operational handles cannot carry an overflowing shape. Exercise the
    // indexing count boundary directly, retaining its stable error contract.
    let err =
        crate::builtins::introspection::object_indexing::selectors::checked_shape_element_count(&[
            usize::MAX,
            2,
        ])
        .unwrap_err();
    assert_eq!(
        err.identifier(),
        Some("RunMat:numArgumentsFromSubscript:InvalidSubstruct")
    );
}

#[test]
fn resident_owner_rejects_overflowing_shape_before_publication() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![usize::MAX, 2],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        );
    assert!(allocation.is_err());
}

#[test]
fn invalid_context_has_stable_identifier() {
    let target = cell(vec![Value::Num(1.0)]);
    let descriptor = substruct(crate::OBJECT_INDEX_PAREN, cell(vec![Value::Num(1.0)]));
    let err = block_on(num_arguments_from_subscript_builtin(
        target,
        descriptor,
        Value::String("Loop".to_string().into()),
    ))
    .expect_err("invalid context should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:numArgumentsFromSubscript:InvalidContext")
    );
}
