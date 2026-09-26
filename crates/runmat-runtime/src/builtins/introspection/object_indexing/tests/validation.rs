use super::*;
#[test]
fn typed_scalar_subscripts_do_not_round_or_saturate() {
    assert_eq!(
        single_subscript_count(
            &Value::Num(0.0),
            &Value::Int(IntValue::U64(u64::MAX)),
            0,
            true
        )
        .expect("positive uint64 subscript"),
        1
    );
    assert!(
        single_subscript_count(&Value::Num(0.0), &Value::Int(IntValue::I64(-1)), 0, true).is_err()
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
            Value::String("Statement".to_string()),
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
    let err = call_err(target, descriptor, Value::String("Statement".to_string()));
    assert_eq!(
        err.identifier(),
        Some("RunMat:numArgumentsFromSubscript:InvalidSubstruct")
    );
}

#[test]
fn default_brace_shape_overflow_returns_substruct_error() {
    let target = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
        shape: vec![usize::MAX, 2],
        device_id: 0,
        buffer_id: 99,
        descriptor: Default::default(),
    });
    let descriptor = substruct(
        crate::OBJECT_INDEX_BRACE,
        cell(vec![Value::String(":".to_string())]),
    );
    let err = call_err(target, descriptor, Value::String("Statement".to_string()));
    assert_eq!(
        err.identifier(),
        Some("RunMat:numArgumentsFromSubscript:InvalidSubstruct")
    );
}

#[test]
fn invalid_context_has_stable_identifier() {
    let target = cell(vec![Value::Num(1.0)]);
    let descriptor = substruct(crate::OBJECT_INDEX_PAREN, cell(vec![Value::Num(1.0)]));
    let err = block_on(num_arguments_from_subscript_builtin(
        target,
        descriptor,
        Value::String("Loop".to_string()),
    ))
    .expect_err("invalid context should fail");
    assert_eq!(
        err.identifier(),
        Some("RunMat:numArgumentsFromSubscript:InvalidContext")
    );
}
